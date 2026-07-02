"""PhotoQuilt SD 2.1 — bootstrap + latent upscale + fixed-tile denoise.

Four phases, identical interface to ``flux1/run_flux1.py``:

  1. Bootstrap at native resolution (generate from ``--prompt`` or encode ``--base-image``).
  2. Bicubic latent upscale to the global ``--height`` x ``--width``.
  3. Inject noise at step ``steps - noise_steps`` on the scheduler's noise ladder.
  4. Denoise the tail in fixed ``--tile-size`` windows (MultiDiffusion scatter-mean
     merge), using ``--tile-prompt`` when set.

SD 2.1 has no chunked-attention transformer, so the tile pass is MultiDiffusion:
every tail step runs the UNet per fixed-size window and averages the per-window
epsilon back onto one shared global latent.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import DDIMScheduler, StableDiffusionPipeline

from tiling import tiled_tail_denoise

DEFAULT_MODEL = "Manojb/stable-diffusion-2-1-base"
VAE_SCALE = 8
DEFAULT_PROMPT = (
    "A mysterious woman stands confidently in elaborate, dark armor adorned with intricate designs, "
    "holding a staff, against a backdrop of smoke and an ominous red sky, with shadowy, gothic "
    "buildings in the distance."
)


def _align(x: int, m: int = VAE_SCALE) -> int:
    return max(m, (int(x) // m) * m)


def _bootstrap_dims(h: int, w: int, long_side: int) -> tuple[int, int]:
    scale = min(1.0, float(long_side) / float(max(h, w)))
    return _align(round(h * scale)), _align(round(w * scale))


@torch.no_grad()
def _encode_image_latent(pipe: StableDiffusionPipeline, img: Image.Image, h: int, w: int, device, dtype) -> torch.Tensor:
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    pixel = pipe.image_processor.preprocess(img, height=h, width=w).to(device=device, dtype=dtype)
    latents = pipe.vae.encode(pixel).latent_dist.sample()
    return latents * pipe.vae.config.scaling_factor


@torch.no_grad()
def _decode_latent(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> Image.Image:
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    return Image.fromarray((image * 255).round().astype("uint8"))


def _encode_prompt(pipe, prompt: str, device, dtype) -> torch.Tensor:
    tokens = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return pipe.text_encoder(tokens.input_ids.to(device))[0].to(dtype=dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mosaic on Stable Diffusion 2.1 (bootstrap + fixed-tile denoise)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for bootstrap generation")
    p.add_argument("--tile-prompt", default=None, help="Prompt for the tiled pass (default: --prompt)")
    p.add_argument("--height", type=int, default=2048, help="Global output height (pixels)")
    p.add_argument("--width", type=int, default=2048, help="Global output width (pixels)")
    p.add_argument("--bootstrap-size", type=int, default=768, metavar="PX", help="Bootstrap long side (default 768)")
    p.add_argument("--tile-size", type=int, default=512, metavar="PX", help="Tile side in pixels (multiple of 8)")
    p.add_argument("--tile-overlap", type=int, default=0, metavar="PX", help="Overlap between tiles in pixels")
    p.add_argument("--steps", type=int, default=50, help="Total denoising steps (bootstrap + tail ladder)")
    p.add_argument("--noise-steps", type=int, default=35, metavar="N", help="Tail steps after injection (inject at steps - N)")
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--base-image", default=None, metavar="PATH", help="Bootstrap from an image instead of generating")
    p.add_argument("--save-bootstrap", action="store_true", help="Also save the bootstrap image before upscale")
    p.add_argument("--tile-batch", type=int, default=4, help="Windows processed per UNet batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace SD 2.1 checkpoint")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.noise_steps < 1:
        raise SystemExit("--noise-steps must be >= 1")
    if args.noise_steps >= args.steps:
        raise SystemExit(f"--noise-steps ({args.noise_steps}) must be < --steps ({args.steps})")
    if args.tile_size % VAE_SCALE != 0:
        raise SystemExit(f"--tile-size must be a multiple of {VAE_SCALE}")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_h, global_w = _align(args.height), _align(args.width)
    boot_h, boot_w = _bootstrap_dims(global_h, global_w, args.bootstrap_size)
    inject_index = args.steps - args.noise_steps
    tile_prompt = args.tile_prompt or args.prompt

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"sd21_mosaic_{global_h}x{global_w}_tile{args.tile_size}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts.txt").write_text(f"prompt: {args.prompt}\ntile: {tile_prompt}\n")

    print(f"Loading {args.model} on {device} …")
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device.type == "cuda":
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()

    scaling = pipe.vae.config.scaling_factor

    # Phase 1: bootstrap (generate or encode base image) → clean latent.
    # Both paths return latents in the UNet/"scaled" convention (encoder output ×
    # scaling_factor); decode with vae.decode(latent / scaling_factor).
    if args.base_image:
        path = Path(args.base_image).expanduser()
        if not path.is_file():
            raise SystemExit(f"--base-image not found: {path}")
        print(f"Bootstrap from image {path} → {boot_w}x{boot_h}, VAE encode")
        boot_latent = _encode_image_latent(pipe, Image.open(path).convert("RGB"), boot_h, boot_w, device, dtype)
    else:
        print(f"Bootstrap generate {boot_w}x{boot_h}, {args.steps} steps")
        gen = torch.Generator(device=device).manual_seed(args.seed)
        boot_latent = pipe(
            args.prompt,
            height=boot_h,
            width=boot_w,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            output_type="latent",
        ).images

    bootstrap_img = None
    if args.save_bootstrap:
        bootstrap_img = _decode_latent(pipe, boot_latent)
    if args.save_bootstrap and bootstrap_img is not None:
        bootstrap_img.save(out_dir / "bootstrap.png")

    # Phase 2: bicubic latent upscale to the global canvas.
    th, tw = global_h // VAE_SCALE, global_w // VAE_SCALE
    clean = F.interpolate(boot_latent.float(), size=(th, tw), mode="bicubic", align_corners=False).to(dtype)
    print(f"Latent upscale {boot_w}x{boot_h} → {global_w}x{global_h} (latent {tw}x{th})")

    # Phase 3: noise injection at the inject step.
    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps
    t_inject = timesteps[inject_index]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(clean.shape, generator=gen, device=device, dtype=clean.dtype)
    latents = pipe.scheduler.add_noise(clean, noise, t_inject)
    print(f"Inject noise at step {inject_index}/{args.steps} (t={int(t_inject.item())}), then {args.noise_steps} tile steps")

    # Phase 4: tiled tail denoise.
    cond = _encode_prompt(pipe, tile_prompt, device, dtype)
    uncond = _encode_prompt(pipe, "", device, dtype)
    tile_latent = args.tile_size // VAE_SCALE
    overlap_latent = args.tile_overlap // VAE_SCALE
    stride_latent = max(1, tile_latent - overlap_latent)

    t0 = time.time()
    latents = tiled_tail_denoise(
        latents=latents,
        timesteps=timesteps,
        inject_index=inject_index,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        cond_embeds=cond,
        uncond_embeds=uncond,
        guidance_scale=args.guidance_scale,
        tile_latent=tile_latent,
        stride_latent=stride_latent,
        tile_batch=args.tile_batch,
    )
    print(f"Tile denoise done in {time.time() - t0:.1f}s")

    image = _decode_latent(pipe, latents)
    out_path = out_dir / "mosaic.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved {out_path} ({global_w}x{global_h})")


if __name__ == "__main__":
    main()
