import unittest

import numpy as np
import torch
from PIL import Image

from sdxl.dino_reservation import border_cost, guide_boundary, share_latents
from sdxl.joint_zoom import board
from sdxl.render_reserved_cycle import origin, resolve_views


class ReservationTest(unittest.TestCase):
    def test_shared_raster_converges_after_one_round_trip(self):
        rng = np.random.default_rng(31)
        templates = [Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)) for _ in range(5)]
        slots = [0, 2, 5, 10, 15]
        views = resolve_views(templates, slots, grid=4)
        for j, view in enumerate(views):
            x, y = origin(slots[j], 16, 4)
            self.assertEqual(view.crop((x, y, x+16, y+16)).tobytes(), views[j-1].reduce(4).tobytes())

    def test_reserved_latent_references_actual_child_states(self):
        z = torch.randn((5, 16, 4, 16, 16), generator=torch.Generator().manual_seed(7)).half()
        slots = [0, 2, 5, 10, 15]
        share_latents(z, slots, 4)
        for j, slot in enumerate(slots):
            child = torch.nn.functional.avg_pool2d(board(z[j-1], 4).float(), 4).half()
            torch.testing.assert_close(z[j, slot:slot+1], child, atol=0, rtol=0)

    def test_boundary_guidance_acts_on_facing_edges_only(self):
        parent = torch.zeros(16, 4, 16, 16)
        following = parent.clone()
        guide_boundary(following, parent, torch.ones_like(parent), 5, 4, 1.)
        self.assertEqual(following.sum().item(), 4*4*16)
        self.assertTrue(following[1, :, -1].eq(1).all())
        self.assertTrue(following[4, :, :, -1].eq(1).all())
        self.assertTrue(following[6, :, :, 0].eq(1).all())
        self.assertTrue(following[9, :, 0].eq(1).all())

    def test_border_score_compares_the_four_neighbors(self):
        rgb = torch.zeros((9, 3, 16, 16))
        rgb[[1, 3, 5, 7]] = 1
        costs = border_cost(rgb, torch.ones_like(rgb[0]), 3)
        self.assertEqual(costs[4].item(), 0)
        self.assertGreater(costs[1].item(), 0)
