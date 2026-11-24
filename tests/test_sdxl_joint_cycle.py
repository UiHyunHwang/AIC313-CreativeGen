import unittest

from PIL import Image
import torch

from sdxl.joint_zoom import guide_pair
from sdxl.render_joint_cycle import FRAMES, frame


class JointCycleTest(unittest.TestCase):
    def test_closing_edge_reads_the_same_clean_estimates_as_the_other_edges(self):
        clean = torch.randn((5, 4, 4, 8, 8), generator=torch.Generator().manual_seed(3))
        def result(order):
            following = clean.clone()
            for j in order:
                guide_pair(following[j], following[j-1], clean[j], clean[j-1], 2, j % 4, .5, 4)
            return following
        torch.testing.assert_close(result(range(5)), result(reversed(range(5))))

    def test_camera_moves_from_child_to_parent_even_for_corner_slots(self):
        child = Image.new("RGB", (300, 300), "red")
        parent = Image.new("RGB", (512, 512), "blue")
        for slot in (0, 15, 240, 255):
            self.assertEqual(frame(child, [parent], slot, 0).getpixel((0,0)), (255,0,0))
            self.assertEqual(frame(child, [parent], slot, FRAMES).getpixel((0,0)), (0,0,255))
