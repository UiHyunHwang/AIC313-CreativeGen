import unittest
from dataclasses import replace

import torch

from sdxl.config import RunConfig
from sdxl.tiling import apply_tile_noise_seeds


class TileNoiseTests(unittest.TestCase):
    def test_selected_tile_matches_standalone_draw(self):
        noise = torch.randn((1, 4, 8, 12), generator=torch.Generator().manual_seed(43))
        original = noise.clone()
        apply_tile_noise_seeds(noise, 4, {"r001_c001": 123})
        expected = torch.randn((1, 4, 4, 4), generator=torch.Generator().manual_seed(123))
        original[..., 4:8, 4:8] = expected
        self.assertTrue(torch.equal(original, noise))

    def test_replay_validates_tile_coordinates_and_seeds(self):
        cfg = RunConfig(prompt="test", height=2048, width=3072, tile_size=1024).normalized()
        replace(cfg, tile_noise_seeds={"r001_c002": 0}).validate()
        for mapping in ({"r002_c000": 1}, {"r000_c003": 0}, {"r0000_c000": 0}, {"r000_c000": True}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                replace(cfg, tile_noise_seeds=mapping).validate()
