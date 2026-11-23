import unittest

import torch

from sdxl.joint_zoom import board, couple, predict_slot, split_board


class JointZoomTest(unittest.TestCase):
    def test_layout_uses_row_major_coordinates(self):
        tiles = torch.arange(16).reshape(4, 1, 2, 2)
        canvas = board(tiles, 2)
        torch.testing.assert_close(canvas[..., :2, 2:], tiles[1:2])
        torch.testing.assert_close(split_board(canvas, 2), tiles)

    def test_coupling_moves_both_scales_and_only_the_reserved_parent_tile(self):
        clean = torch.zeros(8, 4, 8, 8)
        clean[4:] = 1
        following = clean.clone()
        couple(following, clean, grid=2, slot=2, gain=.5)
        torch.testing.assert_close(following[[0, 1, 3]], clean[[0, 1, 3]])
        torch.testing.assert_close(following[2], torch.full_like(clean[2], .425))
        torch.testing.assert_close(following[4:], torch.full_like(clean[4:], .925))
        torch.testing.assert_close(clean[4:], torch.ones_like(clean[4:]))

    def test_diffusion_slot_uses_the_matching_early_prediction(self):
        parent = torch.zeros(4, 4, 8, 8)
        parent[3] = .5
        clean = torch.cat((parent, torch.full_like(parent, .5)))
        self.assertEqual(predict_slot(clean, 2), 3)

    def test_consistent_views_are_a_fixed_point(self):
        tiles = torch.randn((4, 4, 8, 8), generator=torch.Generator().manual_seed(1))
        clean = torch.cat((tiles.clone(), tiles))
        clean[1:2] = torch.nn.functional.avg_pool2d(board(tiles, 2), 2)
        following = clean.clone()
        couple(following, clean, grid=2, slot=1, gain=1)
        torch.testing.assert_close(following, clean)
