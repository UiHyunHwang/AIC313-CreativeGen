import unittest

import torch
from diffusers import DDIMScheduler

from sdxl.hero_cycle import Tail


class HeroCycleScheduleTest(unittest.TestCase):
    def test_paired_schedules_share_the_same_start_and_preserve_the_original_ddim_tail(self):
        scheduler = DDIMScheduler(clip_sample=False, set_alpha_to_one=True)
        scheduler.set_timesteps(13)
        short = Tail(scheduler, 12, device="cpu")
        long = Tail(scheduler, 20, device="cpu")
        torch.testing.assert_close(short.times, scheduler.timesteps[-12:])
        self.assertEqual(long.times[0].item(), short.times[0].item())
        self.assertEqual(long.times[-1].item(), 0)
        self.assertEqual(long.times.unique().numel(), 20)
        gen = torch.Generator().manual_seed(42)
        sample = torch.randn((1, 4, 8, 8), generator=gen)
        prediction = torch.randn(sample.shape, generator=gen)
        for step in short.times:
            expected = scheduler.step(prediction, step, sample, eta=0).prev_sample
            actual, _ = short.step(prediction, step, sample)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
