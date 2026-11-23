import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from safetensors.torch import save_file
import torch

from sdxl.config import RunConfig
from sdxl.run_sdxl_mosaic import _parent_target


class ParentTargetTests(unittest.TestCase):
    def test_verified_latent_crop_and_invalid_sources(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = RunConfig(prompt="outer", height=32, width=48, bootstrap_size=16,
                             tile_size=16, dtype="float32", device="cpu", offload="none",
                             revision="a"*40, scheduler_config={}).normalized()
            (root/"config.json").write_text(json.dumps(base.to_dict()))
            latent = torch.arange(96).float().reshape(1,4,4,6)
            name = "final_latents.safetensors"
            save_file({"latents":latent},root/name)
            sha = hashlib.sha256((root/name).read_bytes()).hexdigest()
            manifest = {"status":"complete","outputs":{name:{"sha256":sha}}}
            (root/"manifest.json").write_text(json.dumps(manifest))
            source = {"run":tmp,"sha256":sha,"row":1,"column":2}
            cfg = replace(base, height=32, width=32, parent_tile_source=source)
            pipe = SimpleNamespace(vae_scale_factor=8,unet=SimpleNamespace(config=SimpleNamespace(in_channels=4)))
            cfg.validate(replay=True)
            torch.testing.assert_close(_parent_target(cfg,pipe), latent[:,:,2:4,4:6])
            for change in [{"column":3},{"sha256":"0"*64}]:
                with self.assertRaises(ValueError):
                    _parent_target(replace(cfg,parent_tile_source=source|change),pipe)
