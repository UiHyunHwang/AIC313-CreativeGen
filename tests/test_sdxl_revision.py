"""Offline model revision resolution; no model loads, network, or CUDA."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sdxl.run_sdxl_mosaic import _resolve_revision


class OfflineRevisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)
        self.snapshots = self.cache / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
        cache_patch = patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache))
        self.addCleanup(cache_patch.stop)
        cache_patch.start()
        ref_patch = patch("huggingface_hub.try_to_load_from_cache", return_value=None)
        self.addCleanup(ref_patch.stop)
        self.lookup = ref_patch.start()
        network_patch = patch("huggingface_hub.model_info", side_effect=AssertionError("Network not allowed"))
        self.addCleanup(network_patch.stop)
        network_patch.start()

    def snapshot(self, commit):
        path = self.snapshots / commit
        path.mkdir(parents=True)
        (path / "model_index.json").write_text("{}")

    def test_single_snapshot_without_main_ref(self):
        self.snapshot("a" * 40)
        self.assertEqual(_resolve_revision(None, True), "a" * 40)

    def test_ambiguous_snapshots_require_explicit_revision(self):
        self.snapshot("a" * 40)
        self.snapshot("b" * 40)
        with self.assertRaisesRegex(RuntimeError, "Multiple cached"):
            _resolve_revision(None, True)

    def test_explicit_missing_branch_does_not_fall_back(self):
        self.snapshot("a" * 40)
        with self.assertRaisesRegex(RuntimeError, "Cannot resolve"):
            _resolve_revision("another-branch", True)

    def test_no_snapshot_reports_error(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot resolve"):
            _resolve_revision(None, True)

    def test_cached_named_ref_takes_precedence(self):
        self.snapshot("a" * 40)
        self.snapshot("b" * 40)
        self.lookup.return_value = str(self.snapshots / ("b" * 40) / "model_index.json")
        self.assertEqual(_resolve_revision(None, True), "b" * 40)


if __name__ == "__main__":
    unittest.main()
