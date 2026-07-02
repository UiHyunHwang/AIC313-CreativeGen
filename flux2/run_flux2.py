"""PhotoQuilt FLUX.2 (Klein) — bootstrap base image + fixed small attention tiles.

Same interface and phases as ``flux1/run_flux1.py``:

  1. Bootstrap at native resolution (generate from ``--prompt`` or encode
     ``--base-image``).
  2. Bicubic latent upscale to the global ``--height`` x ``--width``.
  3. Inject noise at step ``steps - noise_steps`` on the full sigma ladder.
  4. Denoise the tail in fixed ``--tile-size`` attention windows. Each window
     gets its own separated text stream, and (with ``--tile-image-dir``) its
     own gallery image as a FLUX.2-native reference (no Redux needed).

FLUX.2 specifics: Qwen3 text encoder, 4D RoPE ids, batch-norm latent scaling,
``--tile-image-strength`` maps to the reference image's encoded resolution.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_latents

from pipeline_flux2 import Flux2KleinMosaicPipeline
from transformer_flux2 import Flux2Transformer2DModelMosaic, _starts_axis

GPU_MEMORY_THRESHOLD_GB = 60
PATCH_PIXELS = 16  # VAE /8 + 2x2 latent patchify -> one token per 16 px
TILE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DEFAULT_PROMPT = (
    "A mysterious woman stands confidently in elaborate, dark armor adorned with intricate designs, "
    "holding a staff, against a backdrop of smoke and an ominous red sky, with shadowy, gothic "
    "buildings in the distance."
)

CHECKPOINT_SHORTHAND = {
    "klein": "black-forest-labs/FLUX.2-klein-9B",
    "klein-base": "black-forest-labs/FLUX.2-klein-base-9B",
}


def get_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def _align_pixel_dims(height: int, width: int) -> Tuple[int, int]:
    m = PATCH_PIXELS
    return max(m, (int(height) // m) * m), max(m, (int(width) // m) * m)


def _bootstrap_pixel_dims(target_h: int, target_w: int, bootstrap_long_side: int) -> Tuple[int, int]:
    scale = min(1.0, float(bootstrap_long_side) / float(max(target_h, target_w)))
    return _align_pixel_dims(int(round(target_h * scale)), int(round(target_w * scale)))


def _sanitize_token(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9_.-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return (s or "x")[:max_len]


def _gpu_slug() -> str:
    try:
        if not torch.cuda.is_available():
            return "cpu"
        raw = torch.cuda.get_device_name(0).strip().replace(" ", "-")
    except Exception:
        return "gpu-unknown"
    return _sanitize_token(raw, max_len=48)


def build_output_basename(
    *,
    checkpoint_slug: str,
    height: int,
    width: int,
    boot_h: int,
    boot_w: int,
    tile_pixels: int,
    steps: int,
    noise_steps: int,
    inject_step: int,
    seed: int,
    prompt_words: List[str],
    from_image: bool,
    tile_prompt_used: bool,
    gallery_count: int = 0,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ck = _sanitize_token(checkpoint_slug.replace("/", "-"), max_len=56)
    pr = _sanitize_token("_".join(prompt_words) if prompt_words else "prompt")
    src = "img" if from_image else "gen"
    tp = "tp1" if tile_prompt_used else "tp0"
    timg = f"_timg{gallery_count}" if gallery_count > 0 else ""
    body = (
        f"{height}x{width}_bs{boot_h}x{boot_w}_tile{tile_pixels}px"
        f"_ns{noise_steps}_inj{inject_step}_st{steps}_{src}_{tp}{timg}"
    )
    return f"{ts}_{_gpu_slug()}_flux2_mosaic_{ck}_{body}_s{seed}_{pr}.png"


# ── gallery helpers (same semantics as flux1) ────────────────────────────────

def _list_gallery_images(dir_path: str) -> List[Path]:
    root = Path(os.path.expanduser(dir_path))
    if not root.is_dir():
        raise FileNotFoundError(f"--tile-image-dir is not a directory: {root}")
    paths = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in TILE_IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"--tile-image-dir has no images: {root}")
    return paths


def _assign_gallery_to_windows(n_windows: int, n_images: int, seed: int) -> List[int]:
    """Seed-deterministic image index per window; balanced via repeated shuffles."""
    rng = random.Random(seed)
    order: List[int] = []
    while len(order) < n_windows:
        chunk = list(range(n_images))
        rng.shuffle(chunk)
        order.extend(chunk)
    return order[:n_windows]


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _ref_side_from_strength(tile_pixels: int, strength: float) -> int:
    """Map --tile-image-strength to the reference's encoded resolution.

    FLUX.2 conditions on reference image tokens directly (no Redux), so there
    is no token-scale knob; fewer reference tokens = weaker influence.
    """
    base = min(int(tile_pixels), 1024)
    strength = max(0.05, min(1.0, float(strength)))
    side = int(round(base * strength / PATCH_PIXELS)) * PATCH_PIXELS
    return max(64, min(base, side))


def _window_grid(global_h: int, global_w: int, tile_patches: int) -> Tuple[int, int]:
    ph, pw = global_h // PATCH_PIXELS, global_w // PATCH_PIXELS
    return len(_starts_axis(ph, tile_patches)), len(_starts_axis(pw, tile_patches))


@torch.no_grad()
def _encode_tile_gallery(
    pipe: Flux2KleinMosaicPipeline,
    *,
    gallery_paths: List[Path],
    assignment: List[int],
    ref_side: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, int]:
    """Per-window packed reference latents [n_windows, R, 128] + ref token grid."""
    packed: List[torch.Tensor] = []
    for p in gallery_paths:
        img = _center_crop_square(Image.open(p).convert("RGB")).resize(
            (ref_side, ref_side), Image.Resampling.LANCZOS,
        )
        pixel = pipe.image_processor.preprocess(img, height=ref_side, width=ref_side)
        pixel = pixel.to(device=device, dtype=pipe.vae.dtype)
        lat = pipe._encode_vae_image(pixel, generator)        # [1, 128, r, r]
        packed.append(pipe._pack_latents(lat).squeeze(0))     # [r*r, 128]
    ref_grid = ref_side // PATCH_PIXELS
    states = torch.stack([packed[i] for i in assignment], dim=0).to(dtype=dtype)
    print(
        f"[mosaic] Tile gallery: {len(gallery_paths)} images → {len(assignment)} windows, "
        f"{states.shape[1]} reference tokens/window ({ref_side}px refs)",
    )
    return states, ref_grid


# ── latent plumbing ──────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_image_raw_latents(
    pipe: Flux2KleinMosaicPipeline,
    pil_image: Image.Image,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """PIL → raw (denormalized, unpatchified) VAE latents [1, 32, H/8, W/8]."""
    if pil_image.size != (width, height):
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
    pixel = pipe.image_processor.preprocess(pil_image, height=height, width=width)
    pixel = pixel.to(device=device, dtype=pipe.vae.dtype)
    return retrieve_latents(pipe.vae.encode(pixel), generator=generator, sample_mode="argmax")


@torch.no_grad()
def _raw_latents_to_normed(pipe: Flux2KleinMosaicPipeline, raw: torch.Tensor) -> torch.Tensor:
    """Raw [1, 32, h, w] → batch-norm-scaled patchified latents [1, 128, h/2, w/2].

    Kept unpacked: the pipeline's ``prepare_latents`` builds ids and packs.
    """
    patched = pipe._patchify_latents(raw)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patched.device, patched.dtype)
    std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).to(patched.device, patched.dtype)
    return (patched - mean) / std


@torch.no_grad()
def _decode_raw_latents_to_pil(
    pipe: Flux2KleinMosaicPipeline,
    raw: torch.Tensor,
) -> Image.Image:
    decoded = pipe.vae.decode(raw.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


@torch.no_grad()
def _upscale_raw_latents(raw: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        raw.float(), size=(target_h // 8, target_w // 8), mode="bicubic", align_corners=False,
    ).to(raw.dtype)


def _load_pipeline(model_id: str) -> Flux2KleinMosaicPipeline:
    print(f"[mosaic] Loading FLUX.2 pipeline from: {model_id}")
    transformer = Flux2Transformer2DModelMosaic.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    pipe = Flux2KleinMosaicPipeline.from_pretrained(
        model_id, transformer=transformer, torch_dtype=torch.bfloat16,
    )
    pipe.vae.use_tiling = True

    if get_gpu_memory_gb() < GPU_MEMORY_THRESHOLD_GB:
        print(f"[mosaic] CPU offload (GPU {get_gpu_memory_gb():.1f}GB < {GPU_MEMORY_THRESHOLD_GB}GB)")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


@torch.no_grad()
def _prepare_bootstrap_raw_latents(
    pipe: Flux2KleinMosaicPipeline,
    args: argparse.Namespace,
    boot_h: int,
    boot_w: int,
    steps: int,
    guidance: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if args.base_image:
        path = os.path.expanduser(args.base_image)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--base-image not found: {path}")
        img = Image.open(path).convert("RGB")
        print(f"[mosaic] Base image {path} → resize to {boot_w}x{boot_h}, VAE encode")
        return _encode_image_raw_latents(pipe, img, boot_h, boot_w, device, generator)

    print(f"[mosaic] Bootstrap generate {boot_w}x{boot_h}, {steps} steps, prompt (base)")
    generator.manual_seed(args.seed)
    return pipe(
        prompt=args.prompt,
        height=boot_h,
        width=boot_w,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="latent",
        return_dict=False,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mosaic FLUX.2 Klein: bootstrap at native res + fixed-pixel tile denoise",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--tile-prompt", type=str, default=None,
        help="Prompt for the tiled full-res pass (default: same as --prompt)",
    )
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument(
        "--bootstrap-size", type=int, default=1024, metavar="PX",
        help="Bootstrap long side in pixels (aspect matches global; default 1024)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=1024, metavar="PX",
        help="Attention tile side in pixels (multiple of 16; default 1024)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Denoising steps (default: 28 for klein-base, 8 for distilled klein)",
    )
    parser.add_argument(
        "--noise-steps", type=int, default=None, metavar="N",
        help="Tail denoise steps after noise injection "
             "(default: 20 for klein-base, 5 for distilled klein)",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=None,
        help="CFG scale (default: 4.0 for klein-base; 1.0 / disabled for distilled klein)",
    )
    parser.add_argument("--base-image", type=str, default=None, metavar="PATH")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint", type=str, default="klein-base",
        help="'klein', 'klein-base', or any Hugging Face repo ID / local FLUX.2 path",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--tile-image-dir", type=str, default=None, metavar="DIR",
        help="Folder of guidance images: each attention tile is denoised with one "
             "gallery image as a FLUX.2-native reference. Assignment is "
             "deterministic from --seed.",
    )
    parser.add_argument(
        "--tile-image-strength", type=float, default=1.0, metavar="S",
        help="Reference influence in (0, 1]: mapped to the reference's encoded "
             "resolution (fewer reference tokens = weaker guidance)",
    )
    parser.add_argument(
        "--save-bootstrap",
        action="store_true",
        help="Also save the bootstrap image (before latent upscale) under --output-dir",
    )
    args = parser.parse_args()

    if args.tile_size < PATCH_PIXELS or args.tile_size % PATCH_PIXELS != 0:
        parser.error(f"--tile-size must be a positive multiple of {PATCH_PIXELS}")
    if args.bootstrap_size < PATCH_PIXELS:
        parser.error(f"--bootstrap-size must be >= {PATCH_PIXELS}")

    model_id = CHECKPOINT_SHORTHAND.get(args.checkpoint, args.checkpoint)
    ckpt_tag = os.path.basename(model_id.rstrip("/"))

    pipe = _load_pipeline(model_id)
    is_distilled = bool(pipe.config.is_distilled)
    steps = args.steps if args.steps is not None else (8 if is_distilled else 28)
    noise_steps = args.noise_steps if args.noise_steps is not None else (5 if is_distilled else 20)
    guidance = args.guidance_scale if args.guidance_scale is not None else (1.0 if is_distilled else 4.0)
    if noise_steps < 1 or noise_steps >= steps:
        parser.error(f"--noise-steps ({noise_steps}) must satisfy 1 <= noise_steps < steps ({steps})")
    print(
        f"[mosaic] {ckpt_tag}: {'distilled' if is_distilled else 'base'} — "
        f"steps={steps} noise_steps={noise_steps} guidance={guidance}",
    )

    global_h, global_w = _align_pixel_dims(args.height, args.width)
    boot_h, boot_w = _bootstrap_pixel_dims(global_h, global_w, args.bootstrap_size)
    tile_patches = args.tile_size // PATCH_PIXELS
    inject_step = steps - noise_steps
    tile_prompt = args.tile_prompt if args.tile_prompt is not None else args.prompt
    tile_prompt_used = args.tile_prompt is not None

    os.makedirs(args.output_dir, exist_ok=True)
    device = pipe._execution_device
    dtype = next(pipe.transformer.parameters()).dtype
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(gen_device).manual_seed(args.seed)

    # Per-tile gallery references (FLUX.2-native, no Redux)
    ref_states = None
    ref_grid = 0
    gallery_count = 0
    if args.tile_image_dir:
        gallery_paths = _list_gallery_images(args.tile_image_dir)
        gallery_count = len(gallery_paths)
        n_rows, n_cols = _window_grid(global_h, global_w, tile_patches)
        assignment = _assign_gallery_to_windows(n_rows * n_cols, gallery_count, args.seed)
        print(
            f"[mosaic] Tile guidance: {gallery_count} gallery images over a "
            f"{n_rows}x{n_cols} window grid (seed {args.seed})",
        )
        for r in range(n_rows):
            row_names = [gallery_paths[assignment[r * n_cols + c]].name for c in range(n_cols)]
            print(f"[mosaic]   row {r}: {', '.join(row_names)}")
        ref_side = _ref_side_from_strength(args.tile_size, args.tile_image_strength)
        ref_states, ref_grid = _encode_tile_gallery(
            pipe,
            gallery_paths=gallery_paths,
            assignment=assignment,
            ref_side=ref_side,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    # Phase 1+2: bootstrap and latent upscale
    raw_boot = _prepare_bootstrap_raw_latents(
        pipe, args, boot_h, boot_w, steps, guidance, device, generator,
    )
    bootstrap_img = _decode_raw_latents_to_pil(pipe, raw_boot)
    if args.save_bootstrap:
        boot_path = os.path.join(args.output_dir, f"bootstrap_{boot_w}x{boot_h}_s{args.seed}.png")
        bootstrap_img.save(boot_path)
        print(f"[mosaic] Bootstrap image saved to: {boot_path}")

    raw_up = _upscale_raw_latents(raw_boot, global_h, global_w)
    clean_normed = _raw_latents_to_normed(pipe, raw_up).to(device=device, dtype=dtype)

    # Phase 3: noise injection. The sigma shift (mu) is computed at *tile*
    # scale: the tail denoises per window, and FLUX.2's empirical-mu fit
    # degenerates past ~4MP (every sigma collapses to ~1 at 4096px+).
    ph, pw = global_h // PATCH_PIXELS, global_w // PATCH_PIXELS
    window_tokens = min(tile_patches, ph) * min(tile_patches, pw)
    timesteps = pipe.get_mosaic_timesteps(steps, window_tokens, device)
    t_inject = timesteps[inject_step]
    print(
        f"[mosaic] Latent upscale {boot_w}x{boot_h} → {global_w}x{global_h}; "
        f"inject noise at step {inject_step}/{steps} (t≈{float(t_inject):.1f}), "
        f"then {noise_steps} tile steps @ {args.tile_size}px ({tile_patches} patches)",
    )
    generator.manual_seed(args.seed)
    noise = torch.randn(clean_normed.shape, generator=generator, device=device, dtype=dtype)
    noisy_unpacked = pipe.scheduler.scale_noise(clean_normed, t_inject.unsqueeze(0), noise)

    # Phase 4: tiled denoise tail (per-window separated text + optional refs)
    tile_kwargs = dict(
        tile_window_patches=tile_patches,
        tile_grid=(ph, pw),
        tile_ref_states=ref_states,
        tile_ref_grid=ref_grid,
    )
    print(f"[mosaic] Tile pass prompt: {tile_prompt[:80]}{'…' if len(tile_prompt) > 80 else ''}")
    t0 = time.perf_counter()
    image = pipe(
        prompt=tile_prompt,
        height=global_h,
        width=global_w,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        latents=noisy_unpacked,
        denoise_start_step=inject_step,
        tile_kwargs=tile_kwargs,
        mu_image_seq_len=window_tokens,
    ).images[0]
    print(f"[mosaic] Tile denoise done in {time.perf_counter() - t0:.1f}s")

    basename = build_output_basename(
        checkpoint_slug=ckpt_tag,
        height=global_h,
        width=global_w,
        boot_h=boot_h,
        boot_w=boot_w,
        tile_pixels=args.tile_size,
        steps=steps,
        noise_steps=noise_steps,
        inject_step=inject_step,
        seed=args.seed,
        prompt_words=tile_prompt.split()[:3],
        from_image=bool(args.base_image),
        tile_prompt_used=tile_prompt_used,
        gallery_count=gallery_count,
    )
    out_path = os.path.join(args.output_dir, basename)
    image.save(out_path)
    print(f"✓ Image saved to: {out_path}")


if __name__ == "__main__":
    main()
