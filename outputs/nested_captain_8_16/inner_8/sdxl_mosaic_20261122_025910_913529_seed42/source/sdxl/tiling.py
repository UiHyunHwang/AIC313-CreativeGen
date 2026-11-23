"""Independent, non-overlapping SDXL denoising tiles for PhotoQuilt.

The caller supplies a canvas that has already been enlarged and re-noised with
the same DDIM scheduler used here. Each tile sees only its own latent window,
the text condition, and tile-local SDXL size conditioning. Predictions are placed
directly into their canvas regions; no overlap, window averaging, or cross-tile
attention is introduced. One deterministic scheduler update advances the whole
canvas after every tile has been evaluated at the current timestep.

This module is an SDXL adaptation of the PhotoQuilt method, written for the
CreativeGen project. It does not load weights or accept reference images.
"""

from __future__ import annotations

import math

import torch


def apply_tile_noise_seeds(noise: torch.Tensor, tile_side: int, seeds: dict[str, int]) -> None:
    """Replace selected regions of the original CPU noise draw in-place."""
    for key, seed in seeds.items():
        row_text, column_text = key.split("_")
        y = int(row_text[1:]) * tile_side
        x = int(column_text[1:]) * tile_side
        region = noise[..., y:y + tile_side, x:x + tile_side]
        replacement = torch.randn(
            region.shape, generator=torch.Generator(device="cpu").manual_seed(seed),
            dtype=noise.dtype, device="cpu",
        )
        region.copy_(replacement)


def _check_condition(name: str, tensor: torch.Tensor | None, ndim: int) -> None:
    if tensor is None or tensor.ndim != ndim or tensor.shape[0] != 1:
        raise ValueError(f"{name} must be a {ndim}-D tensor with batch size 1")


@torch.inference_mode()
def tiled_tail_denoise(
    *,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    unet,
    scheduler,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    pooled_prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor | None,
    add_time_ids: torch.Tensor,
    negative_add_time_ids: torch.Tensor | None,
    guidance_scale: float,
    tile_latent: int,
    tile_batch: int,
    serial_cfg: bool = False,
    progress: bool = True,
    palette_target: torch.Tensor | None = None,
    palette_strength: float = 0.0,
) -> torch.Tensor:
    """Denoise a single re-noised canvas using fixed square SDXL tiles.

    ``timesteps`` contains only the tail to execute, in sampling order. The
    scheduler must already have its full inference schedule configured; this
    function does not reset or slice it. A DDIM scheduler with ``eta=0`` is
    required by the runner so no random draws occur during this tail.

    Prompt embeddings come from ``StableDiffusionXLPipeline.encode_prompt``.
    Each conditioning tensor has batch size one. ``add_time_ids`` and its
    negative counterpart are shaped ``(1, 6)`` and describe a tile, rather than
    the position of a crop within the large canvas. CFG input order is all
    unconditional samples followed by all conditional samples.
    """
    if latents.ndim != 4 or latents.shape[0] != 1:
        raise ValueError("latents must have shape (1, channels, height, width)")
    if not isinstance(tile_latent, int) or tile_latent < 1:
        raise ValueError("tile_latent must be a positive integer")
    if not isinstance(tile_batch, int) or tile_batch < 1:
        raise ValueError("tile_batch must be a positive integer")
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("guidance_scale must be finite and non-negative")
    height, width = latents.shape[-2:]
    if height < 1 or width < 1 or height % tile_latent or width % tile_latent:
        raise ValueError("latent height and width must be exact multiples of tile_latent")
    if not math.isfinite(palette_strength) or not 0 <= palette_strength <= 1:
        raise ValueError("palette_strength must be between zero and one")
    if (palette_target is None) != (palette_strength == 0):
        raise ValueError("Palette guidance requires a target and a positive strength together")
    if palette_target is not None and palette_target.shape != latents.shape:
        raise ValueError("Palette target must have the same shape as the latent canvas")
    if timesteps.ndim != 1 or len(timesteps) == 0:
        raise ValueError("timesteps must be a non-empty one-dimensional tail schedule")

    _check_condition("prompt_embeds", prompt_embeds, 3)
    _check_condition("pooled_prompt_embeds", pooled_prompt_embeds, 2)
    _check_condition("add_time_ids", add_time_ids, 2)
    if add_time_ids.shape[1] != 6:
        raise ValueError("SDXL base add_time_ids must have shape (1, 6)")

    do_cfg = guidance_scale > 1.0
    if do_cfg:
        _check_condition("negative_prompt_embeds", negative_prompt_embeds, 3)
        _check_condition("negative_pooled_prompt_embeds", negative_pooled_prompt_embeds, 2)
        _check_condition("negative_add_time_ids", negative_add_time_ids, 2)
        if negative_prompt_embeds.shape != prompt_embeds.shape:
            raise ValueError("positive and negative prompt embeddings must have matching shapes")
        if negative_pooled_prompt_embeds.shape != pooled_prompt_embeds.shape:
            raise ValueError("positive and negative pooled embeddings must have matching shapes")
        if negative_add_time_ids.shape != add_time_ids.shape:
            raise ValueError("positive and negative time IDs must have matching shapes")

    coordinates = [
        (y, x)
        for y in range(0, height, tile_latent)
        for x in range(0, width, tile_latent)
    ]
    total_tiles = len(coordinates)

    for step_index, timestep in enumerate(timesteps):
        # Every prediction uses the same unchanged canvas at this timestep.
        noise_canvas = torch.empty_like(latents)
        for start in range(0, total_tiles, tile_batch):
            batch_coordinates = coordinates[start : start + tile_batch]
            windows = torch.cat(
                [
                    latents[:, :, y : y + tile_latent, x : x + tile_latent]
                    for y, x in batch_coordinates
                ],
                dim=0,
            )
            batch_size = len(batch_coordinates)
            if do_cfg and serial_cfg:
                model_input = scheduler.scale_model_input(windows, timestep)
                unconditional = unet(
                    model_input, timestep,
                    encoder_hidden_states=negative_prompt_embeds.expand(batch_size, -1, -1),
                    added_cond_kwargs={
                        "text_embeds": negative_pooled_prompt_embeds.expand(batch_size, -1),
                        "time_ids": negative_add_time_ids.expand(batch_size, -1),
                    }, return_dict=False,
                )[0]
                conditional = unet(
                    model_input, timestep,
                    encoder_hidden_states=prompt_embeds.expand(batch_size, -1, -1),
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds.expand(batch_size, -1),
                        "time_ids": add_time_ids.expand(batch_size, -1),
                    }, return_dict=False,
                )[0]
                prediction = unconditional + guidance_scale * (conditional - unconditional)
                for batch_index, (y, x) in enumerate(batch_coordinates):
                    noise_canvas[:, :, y:y + tile_latent, x:x + tile_latent] = prediction[batch_index:batch_index + 1]
                continue
            if do_cfg:
                model_input = torch.cat((windows, windows), dim=0)
                text = torch.cat(
                    (
                        negative_prompt_embeds.expand(batch_size, -1, -1),
                        prompt_embeds.expand(batch_size, -1, -1),
                    ),
                    dim=0,
                )
                pooled = torch.cat(
                    (
                        negative_pooled_prompt_embeds.expand(batch_size, -1),
                        pooled_prompt_embeds.expand(batch_size, -1),
                    ),
                    dim=0,
                )
                time_ids = torch.cat(
                    (
                        negative_add_time_ids.expand(batch_size, -1),
                        add_time_ids.expand(batch_size, -1),
                    ),
                    dim=0,
                )
            else:
                model_input = windows
                text = prompt_embeds.expand(batch_size, -1, -1)
                pooled = pooled_prompt_embeds.expand(batch_size, -1)
                time_ids = add_time_ids.expand(batch_size, -1)

            model_input = scheduler.scale_model_input(model_input, timestep)
            prediction = unet(
                model_input,
                timestep,
                encoder_hidden_states=text,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                return_dict=False,
            )[0]
            if do_cfg:
                unconditional, conditional = prediction.chunk(2)
                prediction = unconditional + guidance_scale * (conditional - unconditional)

            for batch_index, (y, x) in enumerate(batch_coordinates):
                noise_canvas[:, :, y : y + tile_latent, x : x + tile_latent] = prediction[
                    batch_index : batch_index + 1
                ]

        # A single shared update avoids advancing scheduler state per tile batch.
        update = scheduler.step(noise_canvas, timestep, latents, eta=0.0, return_dict=False)
        latents = update[0]
        if palette_target is not None and step_index >= len(timesteps) // 3:
            # Shift only the four coarse latent means per tile. Leave the first
            # third of the trajectory free to establish the subject's composition.
            next_index = step_index + 1
            alpha = (scheduler.alphas_cumprod[int(timesteps[next_index])]
                     if next_index < len(timesteps) else scheduler.final_alpha_cumprod)
            strength = palette_strength * alpha.sqrt().item()
            for y, x in coordinates:
                region = (..., slice(y, y + tile_latent), slice(x, x + tile_latent))
                delta = palette_target[region].float() - update[1][region].float()
                means = torch.nn.functional.adaptive_avg_pool2d(delta, (2, 2))
                shift = torch.nn.functional.interpolate(
                    means, size=(tile_latent, tile_latent), mode="bilinear", align_corners=False
                )
                latents[region].add_(shift.to(latents.dtype), alpha=strength)
        if progress:
            print(
                f"  tail step {step_index + 1}/{len(timesteps)} "
                f"(t={int(timestep.item())}, {total_tiles} independent tiles)",
                flush=True,
            )

    return latents
