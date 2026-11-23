import unittest

import torch
from diffusers import DDIMScheduler

from sdxl.config import RunConfig
from sdxl.tiling import tiled_tail_denoise


class TileScheduleTests(unittest.TestCase):
    def test_high_start_preserves_twelve_updates(self):
        cfg = RunConfig(prompt="test", steps=20, noise_steps=12, tile_schedule_steps=13).normalized()
        cfg.validate()
        scheduler = DDIMScheduler(timestep_spacing="leading", steps_offset=0)
        scheduler.set_timesteps(cfg.tile_schedule_steps)
        self.assertEqual(scheduler.timesteps[-cfg.noise_steps:].tolist(), list(range(836, -1, -76)))

    def test_palette_is_local_to_each_tile(self):
        class ZeroPrediction:
            def __call__(self, x, timestep, **kwargs):
                return (torch.zeros_like(x),)

        scheduler = DDIMScheduler(clip_sample=False)
        scheduler.set_timesteps(3)
        canvas = torch.zeros(1, 4, 4, 8)
        target = torch.ones_like(canvas)
        target[..., 4:] *= -1
        kwargs = dict(timesteps=scheduler.timesteps, scheduler=scheduler, unet=ZeroPrediction(),
                      prompt_embeds=torch.zeros(1, 2, 3), pooled_prompt_embeds=torch.zeros(1, 3),
                      negative_prompt_embeds=None, negative_pooled_prompt_embeds=None,
                      add_time_ids=torch.zeros(1, 6), negative_add_time_ids=None, guidance_scale=1.,
                      tile_latent=4, tile_batch=1, palette_strength=.25, progress=False)
        full = tiled_tail_denoise(latents=canvas, palette_target=target, **kwargs)
        self.assertTrue((full[..., :4] > 0).all())
        for tile in [slice(0, 4), slice(4, 8)]:
            single = tiled_tail_denoise(latents=canvas[..., tile], palette_target=target[..., tile], **kwargs)
            torch.testing.assert_close(full[..., tile], single)
        torch.testing.assert_close(full[..., :4], -full[..., 4:])
