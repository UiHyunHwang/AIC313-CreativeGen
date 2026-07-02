"""PhotoQuilt FLUX.1 — bootstrap base image + fixed small attention tiles.

  1. Bootstrap at native training resolution (default 1024 long side): generate from
     ``--prompt`` or load ``--base-image`` (resized to bootstrap dims).
  2. Latent bicubic upscale to global ``--height`` x ``--width``.
  3. Inject noise at step ``steps - noise_steps`` on the full sigma ladder.
  4. Denoise the tail in fixed-size windows (``--tile-size`` pixels) with
     ``--tile-prompt`` when set, else ``--prompt``.

RoPE extrapolation is disabled (native training-scale windows only). HSWA chunked
attention is always on for the tile pass. Text tokens use each tile's local origin
for RoPE (``window_local_origin``); global text ids stay at zero origin.

Use ``python -m flux1`` from the repository root, or ``python flux1/run_flux1.py``.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from pipeline_flux1 import Flux1Pipeline, calculate_shift, retrieve_timesteps
from transformer_flux1 import Flux1Transformer2DModel, _starts_axis

GPU_MEMORY_THRESHOLD_GB = 60
VAE_SCALE_FACTOR = 8
PATCH_PIXELS = VAE_SCALE_FACTOR * 2  # 16 px per img_ids patch
BASE_PATCHES = 1024 // PATCH_PIXELS  # 64 at native 1024
NATIVE_LATENT_PATCHES = BASE_PATCHES - 1  # 63
ROPE_SPATIAL_DIM = 56

DEFAULT_PROMPT = (
    "A mysterious woman stands confidently in elaborate, dark armor adorned with intricate designs, "
    "holding a staff, against a backdrop of smoke and an ominous red sky, with shadowy, gothic "
    "buildings in the distance."
)


def get_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def _unwrap_flux_transformer(transformer: torch.nn.Module) -> torch.nn.Module:
    base = getattr(transformer, "base_model", None)
    if base is not None:
        model = getattr(base, "model", None)
        if model is not None:
            return model
    return transformer


def _align_pixel_dims(height: int, width: int, vae_scale_factor: int = VAE_SCALE_FACTOR) -> Tuple[int, int]:
    m = int(vae_scale_factor) * 2
    h = max(m, (int(height) // m) * m)
    w = max(m, (int(width) // m) * m)
    return h, w


def _bootstrap_pixel_dims(
    target_h: int, target_w: int, bootstrap_long_side: int,
) -> Tuple[int, int]:
    """Preserve global aspect; long side = ``bootstrap_long_side``."""
    scale = min(1.0, float(bootstrap_long_side) / float(max(target_h, target_w)))
    return _align_pixel_dims(
        int(round(target_h * scale)),
        int(round(target_w * scale)),
    )


def _pixel_to_latent_hw(height: int, width: int, vae_scale_factor: int = VAE_SCALE_FACTOR) -> Tuple[int, int]:
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))
    return h, w


def _img_ids_extent(height: int, width: int) -> int:
    lh, lw = _pixel_to_latent_hw(height, width)
    return max(lh, lw) // 2


def _packed_latent_seq_len(height: int, width: int, vae_scale_factor: int = VAE_SCALE_FACTOR) -> int:
    h, w = _pixel_to_latent_hw(height, width, vae_scale_factor)
    return (h // 2) * (w // 2)


def pixel_tile_to_patches(tile_pixels: int) -> int:
    """Convert tile side in pixels to img_ids patch count (must be multiple of 16)."""
    if tile_pixels % PATCH_PIXELS != 0:
        raise ValueError(
            f"--tile-size ({tile_pixels}) must be a multiple of {PATCH_PIXELS} "
            f"(Flux patch pixel stride)",
        )
    return tile_pixels // PATCH_PIXELS


def compute_ntk_for_window(window_patches: int) -> float:
    if window_patches <= BASE_PATCHES:
        return 1.0
    exponent = 2 * ROPE_SPATIAL_DIM / (ROPE_SPATIAL_DIM - 2)
    return (window_patches / BASE_PATCHES) ** exponent


def build_flux1_schedule(
    num_steps: int,
    main_pass_indices: List[int],
    tile_patches: int,
) -> List[Tuple[float, Optional[int]]]:
    """Placeholder full schedule; tail indices use fixed ``tile_patches`` windows."""
    ntk = compute_ntk_for_window(tile_patches)
    schedule: List[Tuple[float, Optional[int]]] = [(1.0, None)] * num_steps
    for idx in main_pass_indices:
        schedule[idx] = (ntk, tile_patches)
    return schedule


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
    return f"{ts}_{_gpu_slug()}_flux1_mosaic_{ck}_{body}_s{seed}_{pr}.png"


@torch.no_grad()
def _upscale_packed_latents(
    pipe: Flux1Pipeline,
    packed_low: torch.Tensor,
    boot_h: int,
    boot_w: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    unpacked = pipe._unpack_latents(packed_low, boot_h, boot_w, pipe.vae_scale_factor)
    th, tw = _pixel_to_latent_hw(target_h, target_w, pipe.vae_scale_factor)
    upscaled = torch.nn.functional.interpolate(
        unpacked, size=(th, tw), mode="bicubic", align_corners=False,
    )
    num_channels_latents = pipe.transformer.config.in_channels // 4
    return pipe._pack_latents(upscaled, 1, num_channels_latents, th, tw)


@torch.no_grad()
def _encode_pil_to_packed_latents(
    pipe: Flux1Pipeline,
    pil_image: Image.Image,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if pil_image.size != (width, height):
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
    pixel = pipe.image_processor.preprocess(pil_image, height=height, width=width)
    pixel = pixel.to(device=device, dtype=dtype)
    latents = pipe.vae.encode(pixel).latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4
    h, w = _pixel_to_latent_hw(height, width, pipe.vae_scale_factor)
    return pipe._pack_latents(latents, 1, num_channels_latents, h, w)


@torch.no_grad()
def _decode_packed_latents_to_pil(
    pipe: Flux1Pipeline,
    packed: torch.Tensor,
    height: int,
    width: int,
) -> Image.Image:
    latents = pipe._unpack_latents(packed, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


TILE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _list_gallery_images(dir_path: str) -> List[Path]:
    root = Path(os.path.expanduser(dir_path))
    if not root.is_dir():
        raise FileNotFoundError(f"--tile-image-dir is not a directory: {root}")
    paths = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in TILE_IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(
            f"--tile-image-dir has no images ({'/'.join(sorted(TILE_IMAGE_EXTS))}): {root}",
        )
    return paths


def _window_grid(global_h: int, global_w: int, tile_patches: int) -> Tuple[int, int]:
    """Attention-window grid (rows, cols) — must mirror ``_build_window_index``."""
    lh, lw = _pixel_to_latent_hw(global_h, global_w)
    n_rows = len(_starts_axis(lh // 2, tile_patches))
    n_cols = len(_starts_axis(lw // 2, tile_patches))
    return n_rows, n_cols


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


@torch.no_grad()
def _encode_tile_gallery(
    pipe: Flux1Pipeline,
    *,
    redux_model: str,
    gallery_paths: List[Path],
    assignment: List[int],
    tile_prompt: str,
    strength: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-window conditioning [n_windows, T, joint_dim]: tile prompt + Redux image tokens.

    Window i's tokens are ``cat(text(tile_prompt), redux(gallery[assignment[i]]) * strength)``.
    """
    from diffusers import FluxPriorReduxPipeline

    text_embeds, _, _ = pipe.encode_prompt(
        prompt=tile_prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    text_embeds = text_embeds.to(dtype=dtype)

    print(f"[flux1] Loading Redux image encoder: {redux_model}")
    redux = FluxPriorReduxPipeline.from_pretrained(
        redux_model,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        torch_dtype=torch.bfloat16,
    )
    redux.to(device)

    image_tokens: List[torch.Tensor] = []
    for p in gallery_paths:
        img = _center_crop_square(Image.open(p).convert("RGB"))
        latents = redux.encode_image(img, device, 1)
        tokens = redux.image_embedder(latents).image_embeds  # [1, 729, joint_dim]
        image_tokens.append(tokens.to(dtype=dtype) * float(strength))

    del redux
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    window_states = torch.cat(
        [torch.cat([text_embeds, image_tokens[i]], dim=1) for i in assignment],
        dim=0,
    )
    print(
        f"[flux1] Tile gallery: {len(gallery_paths)} images → {len(assignment)} windows, "
        f"{window_states.shape[1]} cond tokens/window (strength {strength})",
    )
    total_cond = window_states.shape[0] * window_states.shape[1]
    if total_cond > 200_000:
        print(
            f"[flux1] WARNING: {total_cond:,} total conditioning tokens "
            f"({window_states.shape[0]} windows × {window_states.shape[1]}); "
            f"VRAM/runtime grow with window count — consider a larger --tile-size",
        )
    return window_states


def _get_inference_timesteps(
    pipe: Flux1Pipeline,
    num_inference_steps: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    image_seq_len = _packed_latent_seq_len(height, width, pipe.vae_scale_factor)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        image_seq_len,
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, mu=mu)
    return timesteps


@torch.no_grad()
def _noisy_latents_at_step(
    pipe: Flux1Pipeline,
    clean_packed: torch.Tensor,
    step_index: int,
    timesteps: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    t = timesteps[step_index]
    noise = torch.randn(clean_packed.shape, generator=generator, device=device, dtype=dtype)
    clean = clean_packed.to(device=device, dtype=dtype)
    t_batch = t.unsqueeze(0) if t.ndim == 0 else t
    if hasattr(pipe.scheduler, "scale_noise"):
        return pipe.scheduler.scale_noise(clean, t_batch, noise)
    if hasattr(pipe.scheduler, "add_noise"):
        return pipe.scheduler.add_noise(clean, noise, t)
    sigma = float(pipe.scheduler.sigmas[step_index]) if hasattr(pipe.scheduler, "sigmas") else 1.0
    return (1.0 - sigma) * clean + sigma * noise


def install_flux1_hswa(
    pipe: Flux1Pipeline,
    transformer: Flux1Transformer2DModel,
    *,
    schedule: List[Tuple[float, Optional[int]]],
    tile_patches: int,
    img_extent: int,
    overlap_padding: int,
    main_pass_schedule_indices: List[int],
    inject_step: int,
    window_local_origin: bool = True,
    window_encoder_states: Optional[torch.Tensor] = None,
) -> None:
    """Wire fixed-tile HSWA into the pipeline (chunked attention, rope_method=none).

    ``window_encoder_states`` [n_windows, T, joint_dim] replaces the prompt
    conditioning per attention window (group i conditions window i).
    """
    n_text_groups = 1
    flat_cond = txt_ids_cond = None
    if window_encoder_states is not None:
        n_text_groups, t_per, joint_dim = window_encoder_states.shape
        flat_cond = window_encoder_states.reshape(1, n_text_groups * t_per, joint_dim)
        txt_ids_cond = torch.zeros(
            n_text_groups * t_per, 3,
            device=window_encoder_states.device, dtype=torch.float32,
        )

    def _params_at_sched_idx(sched_idx: int) -> Tuple[float, Optional[int]]:
        idx = min(max(0, sched_idx), len(schedule) - 1)
        return schedule[idx]

    def _next_sched_idx_after(loop_step: int) -> int:
        try:
            pos = main_pass_schedule_indices.index(loop_step)
        except ValueError:
            pos = loop_step - main_pass_schedule_indices[0]
        next_pos = min(pos + 1, len(main_pass_schedule_indices) - 1)
        return main_pass_schedule_indices[next_pos]

    def hswa_step_callback(pipeline, step_idx, timestep, callback_kwargs):
        ntk, ws = _params_at_sched_idx(_next_sched_idx_after(step_idx))
        pipeline._hswa_ntk_factor = ntk
        pipeline._hswa_window_size = ws
        return callback_kwargs

    _orig_forward = transformer.forward

    def _patched_forward(*fargs, ntk_factor=None, window_size=None, **fkwargs):
        eff_ntk = getattr(pipe, "_hswa_ntk_factor", ntk_factor)
        eff_ws = getattr(pipe, "_hswa_window_size", window_size)
        if flat_cond is not None:
            fkwargs["encoder_hidden_states"] = flat_cond
            fkwargs["txt_ids"] = txt_ids_cond
        return _orig_forward(
            *fargs,
            window_text_groups=n_text_groups,
            ntk_factor=eff_ntk,
            window_size=eff_ws,
            overlap_padding=overlap_padding,
            rope_method="none",
            spread_min=0.0,
            spread_max=1.0,
            mscale_floor=1.0,
            mscale_ceiling=1.0,
            optimized=True,
            debug_windows=False,
            local_text_pos=False,
            attn_scale_floor=1.0,
            attn_scale_power=0.0,
            max_patches=img_extent - 1,
            window_floor=tile_patches,
            force_global_sega_fft=False,
            window_text_min_frac=None,
            window_local_origin=window_local_origin,
            **fkwargs,
        )

    transformer.forward = _patched_forward
    transformer._hswa_dwt_windows = False
    transformer._hswa_overlap_padding = int(overlap_padding)

    init_sched = main_pass_schedule_indices[0] if main_pass_schedule_indices else inject_step
    pipe._hswa_ntk_factor, pipe._hswa_window_size = _params_at_sched_idx(init_sched)
    pipe._hswa_step_callback = hswa_step_callback


def _load_pipeline(model_id: str, device: Optional[str] = None) -> Flux1Pipeline:
    """Load one full pipeline replica.

    ``device`` pins the whole replica to a specific GPU (used for tile-parallel,
    where each GPU holds an independent replica). When ``device`` is ``None`` the
    single-GPU placement logic applies (CPU offload below the VRAM threshold).
    """
    suffix = f" → {device}" if device else ""
    print(f"[flux1] Loading pipeline from: {model_id}{suffix}")
    transformer = Flux1Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe = Flux1Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.vae.enable_tiling()

    if device is not None:
        pipe.to(device)
        pipe._execution_device = torch.device(device)
    elif get_gpu_memory_gb() < GPU_MEMORY_THRESHOLD_GB:
        print(f"[flux1] CPU offload (GPU {get_gpu_memory_gb():.1f}GB < {GPU_MEMORY_THRESHOLD_GB}GB)")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def _tile_row_split(n_tile_rows: int, num_gpus: int) -> List[Tuple[int, int]]:
    """Contiguous ``[start, end)`` tile-row ranges per GPU, balanced (extras to first GPUs)."""
    base, rem = divmod(n_tile_rows, num_gpus)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(num_gpus):
        count = base + (1 if i < rem else 0)
        ranges.append((start, start + count))
        start += count
    return ranges


def _full_canvas_mu(pipe: Flux1Pipeline, global_h: int, global_w: int) -> float:
    """Sigma-shift ``mu`` for the FULL canvas (matches single-GPU pipeline internals)."""
    seq = _packed_latent_seq_len(global_h, global_w, pipe.vae_scale_factor)
    cfg = pipe.scheduler.config
    return float(
        calculate_shift(
            seq,
            cfg.get("base_image_seq_len", 256),
            seq,
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
    )


@torch.no_grad()
def _generate_tile_parallel(
    pipes: List[Flux1Pipeline],
    *,
    args: argparse.Namespace,
    global_h: int,
    global_w: int,
    tile_patches: int,
    inject_step: int,
    main_pass_indices: List[int],
    tile_prompt: str,
    noisy_latents: torch.Tensor,
) -> Image.Image:
    """Strategy C: split the upscaled latent into bands of whole tile-rows, denoise
    each band on its own GPU replica in parallel (threads), then split the VAE decode
    across the same GPUs and stitch the pixel bands back together.
    """
    num_gpus = len(pipes)
    base = pipes[0]
    num_channels_latents = base.transformer.config.in_channels // 4
    lh, lw = _pixel_to_latent_hw(global_h, global_w, base.vae_scale_factor)
    tile_latent = 2 * tile_patches  # latent rows spanned by one tile
    n_tile_rows = global_h // args.tile_size
    full_mu = _full_canvas_mu(base, global_h, global_w)
    row_ranges = _tile_row_split(n_tile_rows, num_gpus)

    # Slice the (already noise-injected) full latent into per-band latents on GPU 0.
    noisy_unpacked = base._unpack_latents(noisy_latents, global_h, global_w, base.vae_scale_factor)

    print(
        f"[flux1] Tile-parallel: {n_tile_rows} tile-rows across {num_gpus} GPUs "
        f"→ row ranges {row_ranges} (full mu={full_mu:.4f})",
    )

    # Install band-specific HSWA on each replica up front (each replica is used by
    # exactly one band, so this is independent per GPU).
    for gpu_idx, (tr0, tr1) in enumerate(row_ranges):
        band_h = (tr1 - tr0) * args.tile_size
        img_extent = _img_ids_extent(band_h, global_w)
        schedule = build_flux1_schedule(args.steps, main_pass_indices, tile_patches)
        install_flux1_hswa(
            pipes[gpu_idx],
            _unwrap_flux_transformer(pipes[gpu_idx].transformer),
            schedule=schedule,
            tile_patches=tile_patches,
            img_extent=img_extent,
            overlap_padding=args.overlap_padding,
            main_pass_schedule_indices=main_pass_indices,
            inject_step=inject_step,
            window_local_origin=not args.no_window_local_origin,
            window_encoder_states=None,
        )

    @torch.no_grad()  # ThreadPoolExecutor workers don't inherit the caller's grad mode
    def _denoise_band(gpu_idx: int) -> torch.Tensor:
        pipe = pipes[gpu_idx]
        torch.cuda.set_device(gpu_idx)
        dev = torch.device(f"cuda:{gpu_idx}")
        tr0, tr1 = row_ranges[gpu_idx]
        r0, r1 = tr0 * tile_latent, tr1 * tile_latent
        band_lh = r1 - r0
        band_h = (tr1 - tr0) * args.tile_size
        band_packed = pipe._pack_latents(
            noisy_unpacked[:, :, r0:r1, :].to(dev), 1, num_channels_latents, band_lh, lw,
        )
        gen = torch.Generator(dev).manual_seed(args.seed)
        band_latent = pipe(
            prompt=tile_prompt,
            height=band_h,
            width=global_w,
            guidance_scale=4.5,
            generator=gen,
            num_inference_steps=args.steps,
            latents=band_packed,
            denoise_start_step=inject_step,
            mu_override=full_mu,
            output_type="latent",
            return_dict=False,
            callback_on_step_end=pipe._hswa_step_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )[0]
        unpacked = pipe._unpack_latents(band_latent, band_h, global_w, pipe.vae_scale_factor)
        return unpacked.to("cuda:0")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_gpus) as ex:
        band_latents = list(ex.map(_denoise_band, range(num_gpus)))
    full_latent = torch.cat(band_latents, dim=2)  # concat along latent height
    del band_latents, noisy_unpacked
    print(f"[flux1] Tile-parallel denoise done in {time.perf_counter() - t0:.1f}s")

    # The transformer + text encoders are no longer needed (denoise is finished),
    # but each replica still pins ~30-40 GB per GPU: the diffusion weights plus the
    # chunked-attention processors/buffers cached on the transformer during the band
    # passes. Just dropping the Python references does NOT free them — the HSWA
    # monkey-patched ``forward`` closure and the per-block chunked processors keep the
    # module alive, so the GPU storage survives gc + empty_cache(). Instead we (a) tear
    # down the chunked processors so their cached index/rope tensors are released, and
    # (b) physically relocate the weights to CPU with ``.to("cpu")``, which frees the
    # GPU storage in place regardless of any lingering references. The VAE stays on GPU
    # and then owns essentially the whole device for decoding.
    for i, pipe in enumerate(pipes):
        torch.cuda.set_device(i)
        tf = _unwrap_flux_transformer(pipe.transformer)
        if getattr(tf, "_chunked_proc_active", False):
            tf._restore_standard_processors()
        if hasattr(tf, "_mask_cache"):
            tf._mask_cache.clear()
        pipe.transformer.to("cpu")
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
    gc.collect()
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    mem = ", ".join(
        f"cuda:{i} {torch.cuda.memory_allocated(i) / 1024**3:.1f}GB"
        for i in range(num_gpus)
    )
    print(f"[flux1] Freed transformer/text-encoders before decode; allocated now: {mem}")

    # ── 2-D blocked VAE decode across the GPUs ───────────────────────────────
    # An earlier version decoded full-canvas-WIDTH strips, but diffusers'
    # ``tiled_decode`` accumulates every internal tile across the strip's full
    # width in a Python list (full-resolution pixels, fp32 inside the decoder)
    # before concatenating — so peak memory scaled with the canvas WIDTH no matter
    # how thin the strip was, and an 8192px-wide strip alone exhausted an 80 GB
    # GPU. We instead decode bounded 2-D latent blocks (height AND width capped at
    # ``--vae-decode-chunk-tile-rows`` tile-units), so peak memory is independent
    # of the total canvas size in both dimensions. Each block carries an overlap
    # margin (``--vae-decode-overlap`` latent rows/cols) into its neighbours so the
    # VAE receptive field has cross-seam context; the margin is cropped afterwards.
    # Blocks are distributed round-robin across GPUs, decoded sequentially per GPU,
    # and moved to CPU (PIL) immediately so VRAM never accumulates.
    ov = max(0, int(args.vae_decode_overlap))
    blk = max(1, int(args.vae_decode_chunk_tile_rows)) * tile_latent  # block size in latent units
    n_tile_cols = global_w // args.tile_size
    lw_total = n_tile_cols * tile_latent  # == lw
    row_starts = list(range(0, lh, blk))
    col_starts = list(range(0, lw_total, blk))
    blocks = [(rs, cs) for rs in row_starts for cs in col_starts]

    @torch.no_grad()  # ThreadPoolExecutor workers don't inherit the caller's grad mode
    def _decode_block(gpu_idx: int, rs: int, cs: int) -> Tuple[int, int, Image.Image]:
        pipe = pipes[gpu_idx]
        torch.cuda.set_device(gpu_idx)
        dev = torch.device(f"cuda:{gpu_idx}")
        re_, ce = min(rs + blk, lh), min(cs + blk, lw_total)
        ds, de = max(0, rs - ov), min(lh, re_ + ov)        # padded latent rows
        ls, le = max(0, cs - ov), min(lw_total, ce + ov)   # padded latent cols
        sub = full_latent[:, :, ds:de, ls:le].to(dev)
        sub = (sub / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(sub, return_dict=False)[0].detach()
        img = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        vsf = pipe.vae_scale_factor
        top, left = (rs - ds) * vsf, (cs - ls) * vsf
        keep_h, keep_w = (re_ - rs) * vsf, (ce - cs) * vsf
        cropped = img.crop((left, top, left + keep_w, top + keep_h))
        del sub, decoded
        torch.cuda.empty_cache()
        return rs * vsf, cs * vsf, cropped

    def _decode_worker(gpu_idx: int) -> List[Tuple[int, int, Image.Image]]:
        # This GPU owns block indices gpu_idx, gpu_idx+num_gpus, … (sequential here).
        return [
            _decode_block(gpu_idx, *blocks[bi])
            for bi in range(gpu_idx, len(blocks), num_gpus)
        ]

    t1 = time.perf_counter()
    print(
        f"[flux1] Tile-parallel VAE decode: {len(blocks)} blocks "
        f"({len(row_starts)}x{len(col_starts)} grid, <= {blk} latent units each) "
        f"across {num_gpus} GPUs (overlap {ov} latent rows/cols)",
    )
    with ThreadPoolExecutor(max_workers=num_gpus) as ex:
        per_gpu = list(ex.map(_decode_worker, range(num_gpus)))
    print(f"[flux1] Tile-parallel VAE decode done in {time.perf_counter() - t1:.1f}s")

    final = Image.new("RGB", (global_w, global_h))
    for results in per_gpu:
        for py, px, block_img in results:
            final.paste(block_img, (px, py))
    return final


@torch.no_grad()
def _prepare_bootstrap_latents(
    pipe: Flux1Pipeline,
    args: argparse.Namespace,
    boot_h: int,
    boot_w: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if args.base_image:
        path = os.path.expanduser(args.base_image)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--base-image not found: {path}")
        img = Image.open(path).convert("RGB")
        print(f"[flux1] Base image {path} → resize to {boot_w}x{boot_h}, VAE encode")
        return _encode_pil_to_packed_latents(pipe, img, boot_h, boot_w, device, dtype)

    print(f"[flux1] Bootstrap generate {boot_w}x{boot_h}, {args.steps} steps, prompt (base)")
    generator.manual_seed(args.seed)
    return pipe(
        prompt=args.prompt,
        height=boot_h,
        width=boot_w,
        guidance_scale=4.5,
        generator=generator,
        num_inference_steps=args.steps,
        output_type="latent",
        return_dict=False,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mosaic FLUX.1: bootstrap at native res + fixed-pixel tile denoise",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt for bootstrap generation")
    parser.add_argument(
        "--tile-prompt",
        type=str,
        default=None,
        help="Prompt for the tiled full-res pass (default: same as --prompt)",
    )
    parser.add_argument("--height", type=int, default=4096, help="Global output height (pixels)")
    parser.add_argument("--width", type=int, default=4096, help="Global output width (pixels)")
    parser.add_argument(
        "--bootstrap-size",
        type=int,
        default=1024,
        metavar="PX",
        help="Bootstrap long side in pixels (aspect matches global; default 1024)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        metavar="PX",
        help="Attention tile side in pixels (multiple of 16; default 1024)",
    )
    parser.add_argument("--steps", type=int, default=28, help="Denoising steps (bootstrap + tail ladder)")
    parser.add_argument(
        "--noise-steps",
        type=int,
        default=20,
        metavar="N",
        help="Tail denoise steps after noise injection (inject at steps - N)",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=None,
        metavar="PATH",
        help="Use an existing image as bootstrap base (resized to bootstrap dims, then encode)",
    )
    parser.add_argument(
        "--save-bootstrap",
        action="store_true",
        help="Also save the bootstrap image (before latent upscale) as *_bootstrap.png",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="Krea-dev")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--overlap-padding", type=int, default=0, help="Extra patch padding on tile edges")
    parser.add_argument(
        "--no-window-local-origin",
        action="store_true",
        help="Disable per-tile text RoPE at window origin (use global zero-origin text ids)",
    )
    parser.add_argument(
        "--tile-image-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Folder of guidance images for the tile pass: each attention tile is "
             "denoised conditioned on one gallery image (Redux tokens + tile prompt). "
             "Images are assigned to tiles deterministically from --seed.",
    )
    parser.add_argument(
        "--tile-image-strength",
        type=float,
        default=1.0,
        metavar="S",
        help="Scale on Redux image tokens for tile guidance (default 1.0; lower = weaker)",
    )
    parser.add_argument(
        "--redux-model",
        type=str,
        default="black-forest-labs/FLUX.1-Redux-dev",
        help="Redux image-encoder checkpoint used with --tile-image-dir",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Tile-parallel generation: replicate the model on every visible GPU, split the "
             "upscaled latent into bands of whole tile-rows, denoise each band on its own GPU "
             "in parallel, then split the VAE decode across GPUs. Requires a tile-aligned canvas "
             "(height and width exact multiples of --tile-size) and >= num_gpus tile-rows.",
    )
    parser.add_argument(
        "--vae-decode-overlap",
        type=int,
        default=8,
        metavar="N",
        help="Latent-row/col overlap between blocks during tile-parallel VAE decode (cropped "
             "after decode) to suppress seams (default 8 = 64 px).",
    )
    parser.add_argument(
        "--vae-decode-chunk-tile-rows",
        type=int,
        default=4,
        metavar="N",
        help="Tile-parallel VAE decode block size in tile-units, bounding BOTH height and width "
             "(default 4 → 4x4 tiles per block). Lower this if the final decode runs out of "
             "memory at very large canvases; blocks are distributed round-robin across GPUs and "
             "decoded sequentially per GPU to bound peak VRAM independent of canvas size.",
    )
    args = parser.parse_args()

    if args.noise_steps < 1:
        parser.error("--noise-steps must be >= 1")
    if args.noise_steps >= args.steps:
        parser.error(f"--noise-steps ({args.noise_steps}) must be < --steps ({args.steps})")
    if args.bootstrap_size < PATCH_PIXELS:
        parser.error(f"--bootstrap-size must be >= {PATCH_PIXELS}")
    if args.tile_size < PATCH_PIXELS:
        parser.error(f"--tile-size must be >= {PATCH_PIXELS}")
    if args.tile_size % PATCH_PIXELS != 0:
        parser.error(f"--tile-size must be a multiple of {PATCH_PIXELS}")

    global_h, global_w = _align_pixel_dims(args.height, args.width)
    boot_h, boot_w = _bootstrap_pixel_dims(global_h, global_w, args.bootstrap_size)
    tile_patches = pixel_tile_to_patches(args.tile_size)
    inject_step = args.steps - args.noise_steps
    main_pass_indices = list(range(inject_step, args.steps))
    tile_prompt = args.tile_prompt if args.tile_prompt is not None else args.prompt
    tile_prompt_used = args.tile_prompt is not None

    # ── Tile-parallel (multi-GPU) feasibility checks ─────────────────────────
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = bool(args.multi_gpu) and num_gpus >= 2
    if args.multi_gpu and num_gpus < 2:
        print(f"[flux1] --multi-gpu ignored: only {num_gpus} GPU(s) visible; running single-GPU")
    if use_multi_gpu:
        if args.tile_image_dir:
            parser.error("--multi-gpu does not support --tile-image-dir (gallery) yet")
        if global_h % args.tile_size != 0 or global_w % args.tile_size != 0:
            parser.error(
                f"--multi-gpu requires a tile-aligned canvas: height ({global_h}) and width "
                f"({global_w}) must be exact multiples of --tile-size ({args.tile_size})",
            )
        n_tile_rows = global_h // args.tile_size
        if n_tile_rows < num_gpus:
            parser.error(
                f"--multi-gpu needs at least num_gpus tile-rows: {n_tile_rows} rows "
                f"(height {global_h} / tile {args.tile_size}) < {num_gpus} GPUs",
            )

    os.makedirs(args.output_dir, exist_ok=True)

    shorthand = {"dev", "Krea-dev"}
    model_id = (
        f"black-forest-labs/FLUX.1-{args.checkpoint}"
        if args.checkpoint in shorthand
        else args.checkpoint
    )
    ckpt_tag = os.path.basename(model_id.rstrip("/"))

    if use_multi_gpu:
        print(f"[flux1] Tile-parallel: loading {num_gpus} replicas (one per GPU)")
        pipes = [_load_pipeline(model_id, device=f"cuda:{i}") for i in range(num_gpus)]
    else:
        pipes = [_load_pipeline(model_id)]
    pipe = pipes[0]
    gen_device = "cuda" if getattr(pipe, "_execution_device", torch.device("cuda")).type == "cuda" else "cpu"
    generator = torch.Generator(gen_device).manual_seed(args.seed)
    device = pipe._execution_device
    dtype = next(pipe.transformer.parameters()).dtype

    window_states = None
    gallery_count = 0
    if args.tile_image_dir:
        gallery_paths = _list_gallery_images(args.tile_image_dir)
        gallery_count = len(gallery_paths)
        n_rows, n_cols = _window_grid(global_h, global_w, tile_patches)
        assignment = _assign_gallery_to_windows(n_rows * n_cols, gallery_count, args.seed)
        print(
            f"[flux1] Tile guidance: {gallery_count} gallery images over a "
            f"{n_rows}x{n_cols} window grid (seed {args.seed})",
        )
        for r in range(n_rows):
            row_names = [gallery_paths[assignment[r * n_cols + c]].name for c in range(n_cols)]
            print(f"[flux1]   row {r}: {', '.join(row_names)}")
        window_states = _encode_tile_gallery(
            pipe,
            redux_model=args.redux_model,
            gallery_paths=gallery_paths,
            assignment=assignment,
            tile_prompt=tile_prompt,
            strength=args.tile_image_strength,
            device=device,
            dtype=dtype,
        )

    bootstrap_packed = _prepare_bootstrap_latents(
        pipe, args, boot_h, boot_w, device, dtype, generator,
    )
    bootstrap_img: Optional[Image.Image] = None
    if args.save_bootstrap:
        bootstrap_img = _decode_packed_latents_to_pil(pipe, bootstrap_packed, boot_h, boot_w)
    if args.save_bootstrap and bootstrap_img is not None:
        prompt_words_boot = args.prompt.split()[:3]
        boot_basename = build_output_basename(
            checkpoint_slug=ckpt_tag,
            height=global_h,
            width=global_w,
            boot_h=boot_h,
            boot_w=boot_w,
            tile_pixels=args.tile_size,
            steps=args.steps,
            noise_steps=args.noise_steps,
            inject_step=inject_step,
            seed=args.seed,
            prompt_words=prompt_words_boot,
            from_image=bool(args.base_image),
            tile_prompt_used=tile_prompt_used,
            gallery_count=gallery_count,
        ).removesuffix(".png") + "_bootstrap.png"
        boot_path = os.path.join(args.output_dir, boot_basename)
        bootstrap_img.save(boot_path)
        print(f"[flux1] Bootstrap image saved to: {boot_path}")
    clean_packed = _upscale_packed_latents(
        pipe, bootstrap_packed, boot_h, boot_w, global_h, global_w,
    ).to(device=device, dtype=dtype)

    if args.base_image:
        target_img = Image.open(os.path.expanduser(args.base_image)).convert("RGB").resize(
            (global_w, global_h), Image.Resampling.LANCZOS,
        )
    else:
        target_img = _decode_packed_latents_to_pil(pipe, clean_packed, global_h, global_w)

    timesteps_full = _get_inference_timesteps(pipe, args.steps, global_h, global_w, device)
    noise_t = float(timesteps_full[inject_step].item())
    print(
        f"[flux1] Latent upscale {boot_w}x{boot_h} → {global_w}x{global_h}; "
        f"inject noise at step {inject_step}/{args.steps} (t≈{noise_t:.1f}), "
        f"then {args.noise_steps} tile steps @ {args.tile_size}px ({tile_patches} patches)",
    )

    generator.manual_seed(args.seed)
    noisy_latents = _noisy_latents_at_step(
        pipe, clean_packed, inject_step, timesteps_full, device, dtype, generator,
    )

    print(f"[flux1] Tile pass prompt: {tile_prompt[:80]}{'…' if len(tile_prompt) > 80 else ''}")
    if use_multi_gpu:
        image = _generate_tile_parallel(
            pipes,
            args=args,
            global_h=global_h,
            global_w=global_w,
            tile_patches=tile_patches,
            inject_step=inject_step,
            main_pass_indices=main_pass_indices,
            tile_prompt=tile_prompt,
            noisy_latents=noisy_latents,
        )
    else:
        img_extent = _img_ids_extent(global_h, global_w)
        schedule = build_flux1_schedule(args.steps, main_pass_indices, tile_patches)
        install_flux1_hswa(
            pipe,
            _unwrap_flux_transformer(pipe.transformer),
            schedule=schedule,
            tile_patches=tile_patches,
            img_extent=img_extent,
            overlap_padding=args.overlap_padding,
            main_pass_schedule_indices=main_pass_indices,
            inject_step=inject_step,
            window_local_origin=not args.no_window_local_origin,
            window_encoder_states=window_states,
        )
        t0 = time.perf_counter()
        image = pipe(
            prompt=tile_prompt,
            height=global_h,
            width=global_w,
            guidance_scale=4.5,
            generator=generator,
            num_inference_steps=args.steps,
            latents=noisy_latents,
            denoise_start_step=inject_step,
            callback_on_step_end=pipe._hswa_step_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
        print(f"[flux1] Tile denoise done in {time.perf_counter() - t0:.1f}s")

    prompt_words = tile_prompt.split()[:3]
    basename = build_output_basename(
        checkpoint_slug=ckpt_tag,
        height=global_h,
        width=global_w,
        boot_h=boot_h,
        boot_w=boot_w,
        tile_pixels=args.tile_size,
        steps=args.steps,
        noise_steps=args.noise_steps,
        inject_step=inject_step,
        seed=args.seed,
        prompt_words=prompt_words,
        from_image=bool(args.base_image),
        tile_prompt_used=tile_prompt_used,
        gallery_count=gallery_count,
    )
    out_path = os.path.join(args.output_dir, basename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path)
    print(f"✓ Image saved to: {out_path}")


if __name__ == "__main__":
    main()
