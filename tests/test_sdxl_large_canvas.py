"""CPU regressions for large-canvas memory optimizations; no weights or CUDA."""

import unittest
from types import SimpleNamespace

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from sdxl.decode import decode_tiles_to_cpu
from sdxl.tiling import tiled_tail_denoise


class ToyVAE:
    tile_latent_min_size = 4
    tile_sample_min_size = 8
    tile_overlap_factor = 0.25
    config = SimpleNamespace(use_post_quant_conv=False)
    blend_v = AutoencoderKL.blend_v
    blend_h = AutoencoderKL.blend_h

    def decoder(self, block):
        # Depend on block-local content, so overlaps differ and exercise blend.
        return torch.nn.functional.interpolate(block, scale_factor=2, mode="nearest") + block.mean()

    def decode(self, block, return_dict=False):
        if max(block.shape[-2:]) > self.tile_latent_min_size:
            raise AssertionError("Attempted full-canvas decode")
        return (self.decoder(block),)


class LargeCanvasTests(unittest.TestCase):
    def test_cpu_accumulation_matches_official_tiled_decode(self):
        generator = torch.Generator().manual_seed(7)
        for shape in [(1, 3, 12, 12), (1, 3, 13, 17), (1, 3, 4, 4)]:
            with self.subTest(shape=shape):
                latents = torch.randn(shape, generator=generator)
                vae = ToyVAE()
                expected = AutoencoderKL.tiled_decode(vae, latents, return_dict=False)[0]
                actual = decode_tiles_to_cpu(vae, latents)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_serial_cfg_matches_batched_cfg(self):
        class Scheduler:
            def scale_model_input(self, value, timestep):
                return value * 0.9

            def step(self, prediction, timestep, value, **kwargs):
                return (value - prediction * 0.1,)

        class UNet:
            def __init__(self):
                self.batches = []

            def __call__(self, value, timestep, encoder_hidden_states, added_cond_kwargs, **kwargs):
                self.batches.append(value.shape[0])
                text = encoder_hidden_states.mean(dim=(1, 2))
                pooled = added_cond_kwargs["text_embeds"].mean(dim=1)
                time = added_cond_kwargs["time_ids"].mean(dim=1)
                return (value * 0.2 + (text + pooled + time).view(-1, 1, 1, 1),)

        arguments = dict(
            latents=torch.arange(256, dtype=torch.float32).reshape(1, 4, 8, 8) / 256,
            timesteps=torch.tensor([3, 2]), scheduler=Scheduler(),
            prompt_embeds=torch.ones(1, 2, 3), negative_prompt_embeds=torch.zeros(1, 2, 3),
            pooled_prompt_embeds=torch.ones(1, 3) * 2, negative_pooled_prompt_embeds=torch.ones(1, 3) * -1,
            add_time_ids=torch.ones(1, 6) * 4, negative_add_time_ids=torch.ones(1, 6),
            guidance_scale=5.0, tile_latent=4, tile_batch=1, progress=False,
        )
        batched, serial = UNet(), UNet()
        expected = tiled_tail_denoise(**arguments, unet=batched)
        actual = tiled_tail_denoise(**arguments, unet=serial, serial_cfg=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(set(batched.batches), {2})
        self.assertEqual(set(serial.batches), {1})


if __name__ == "__main__":
    unittest.main()
