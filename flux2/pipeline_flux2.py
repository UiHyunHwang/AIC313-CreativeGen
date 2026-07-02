"""FLUX.2 Klein pipeline with partial-denoise support for mosaic generation.

Subclasses diffusers' ``Flux2KleinPipeline``:
- ``denoise_start_step``: skip the first steps of the sigma ladder and denoise a
  pre-noised ``latents`` tensor for the remaining tail (mosaic noise injection).
- ``tile_kwargs``: forwarded to the transformer on every call (cond and uncond),
  activating window attention with per-tile text/reference conditioning.
- ``mu_image_seq_len``: sequence length used for the empirical sigma shift.
  FLUX.2's ``compute_empirical_mu`` is fitted up to ~4MP and explodes beyond it
  (exp(mu) ≈ 1e5 at 4096px collapses every sigma to ~1). The tile pass denoises
  per window, so the shift must be computed at tile scale, not canvas scale.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2KleinPipeline,
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput


class Flux2KleinMosaicPipeline(Flux2KleinPipeline):

    def get_mosaic_timesteps(self, num_inference_steps: int, image_seq_len: int, device):
        """Full sigma ladder identical to the one ``__call__`` will build."""
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        return timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
        denoise_start_step: int = 0,
        tile_kwargs: dict[str, Any] | None = None,
        mu_image_seq_len: int | None = None,
    ):
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=["latents"],
            guidance_scale=guidance_scale,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False

        denoise_start_step = max(0, int(denoise_start_step))
        if denoise_start_step > 0 and latents is None:
            raise ValueError("denoise_start_step > 0 requires pre-noised `latents`")

        batch_size = 1 if (prompt is None or isinstance(prompt, str)) else len(prompt)
        device = self._execution_device

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        if self.do_classifier_free_guidance:
            negative_prompt = "" if (prompt is None or isinstance(prompt, str)) else [""] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        timesteps = self.get_mosaic_timesteps(
            num_inference_steps,
            image_seq_len=mu_image_seq_len if mu_image_seq_len is not None else latents.shape[1],
            device=device,
        )
        if denoise_start_step >= len(timesteps):
            raise ValueError(
                f"denoise_start_step={denoise_start_step} must be < {len(timesteps)} steps",
            )
        self._num_timesteps = len(timesteps)
        self.scheduler.set_begin_index(denoise_start_step)
        tile_kwargs = tile_kwargs or {}

        with self.progress_bar(total=len(timesteps) - denoise_start_step) as progress_bar:
            for i, t in enumerate(timesteps):
                if i < denoise_start_step or self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(self.transformer.dtype)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                        **tile_kwargs,
                    )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_ids,
                            return_dict=False,
                            **tile_kwargs,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_outputs = callback_on_step_end(self, i, t, {"latents": latents})
                    if callback_outputs:
                        latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        self._current_timestep = None

        latents = self._unpack_latents_with_ids(latents, latent_ids)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return Flux2PipelineOutput(images=image)
