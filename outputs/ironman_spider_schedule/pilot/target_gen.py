from dataclasses import replace
from pathlib import Path
import json
import torch
from safetensors.torch import save_file
from sdxl.config import RunConfig
from sdxl.run_sdxl_mosaic import _configure_determinism, _load_pipeline, _decode_latent
from sdxl.memory_guard import GpuMemoryGuard

out = Path(__file__).resolve().parent
old = RunConfig.read(Path("outputs/sdxl_ironman_spiderman_short_8x8_20261119_050359/recipe.json"))
cfg = replace(old, prompt="A front-facing portrait of Iron Man fills the frame with a bold red-and-gold helmet, bright white eyes, and a plain dark background.", tile_prompt="spider man")
(out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
with GpuMemoryGuard(3, out), torch.inference_mode():
    _configure_determinism(True)
    pipe = _load_pipeline(cfg, True)
    z = pipe(prompt=cfg.prompt, negative_prompt="", height=1024, width=1024, original_size=(1024,1024), target_size=(1024,1024), num_inference_steps=20, guidance_scale=5, eta=0, generator=torch.Generator(device="cpu").manual_seed(42), output_type="latent").images
    save_file({"latents": z.cpu().contiguous()}, out / "target_latents.safetensors")
    _decode_latent(pipe,z).save(out / "target.png")
