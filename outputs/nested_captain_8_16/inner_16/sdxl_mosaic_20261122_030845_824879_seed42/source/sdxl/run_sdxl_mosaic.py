"""Generate a PhotoQuilt target and mosaic with the official SDXL base only.

This local SDXL implementation follows PhotoQuilt (arXiv:2606.30968v2).
It generates the target internally; no reference-image or add-on-weight loader
is exposed. A saved recipe always regenerates both target and mosaic.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MODEL_ID, PROJECT_ROOT, RunConfig, parse_options
from .memory_guard import GpuMemoryGuard, MemoryGuardStop


RUNTIME_PACKAGES = (
    "torch", "diffusers", "transformers", "accelerate", "huggingface-hub",
    "safetensors", "numpy", "Pillow", "tokenizers", "ftfy", "regex",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_source(out_dir: Path) -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "sdxl").glob("*.py"))
    paths += [PROJECT_ROOT / "sdxl" / "README.md", PROJECT_ROOT / "requirements-sdxl.txt", PROJECT_ROOT / "LICENSE"]
    hashes = {}
    for path in paths:
        if path.is_file():
            relative = path.relative_to(PROJECT_ROOT)
            destination = out_dir / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            hashes[str(relative)] = _file_hash(destination)
    return hashes


def _git_provenance() -> dict[str, Any]:
    def read(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), *arguments],
                capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"head": read("rev-parse", "HEAD"), "status": read("status", "--short")}


def _resolve_revision(revision: str | None, local_files_only: bool) -> str:
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    if local_files_only:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(MODEL_ID, "model_index.json", revision=revision or "main")
        if isinstance(cached, str):
            commit = Path(cached).parent.name
            if re.fullmatch(r"[0-9a-f]{40}", commit):
                return commit
        if revision is None:
            # Downloading by commit SHA need not create refs/main. An offline
            # cache can therefore hold all weights without a main ref. Select
            # only an unambiguous snapshot; never guess between model versions.
            from huggingface_hub.constants import HF_HUB_CACHE

            snapshots = Path(HF_HUB_CACHE) / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
            candidates = sorted(
                path.name for path in snapshots.glob("*")
                if path.is_dir() and re.fullmatch(r"[0-9a-f]{40}", path.name)
                and (path / "model_index.json").is_file()
            )
            if len(candidates) == 1:
                print(f"Using the sole cached official model revision: {candidates[0]}", flush=True)
                return candidates[0]
            if len(candidates) > 1:
                raise RuntimeError(
                    "Multiple cached model revisions exist without a main ref. "
                    "Choose one explicitly with --revision: " + ", ".join(candidates)
                )
        raise RuntimeError(
            "Cannot resolve an official cached model revision. Cache the model first, "
            "or provide its exact --revision SHA together with --local-files-only."
        )
    from huggingface_hub import model_info

    commit = model_info(MODEL_ID, revision=revision).sha
    if commit is None or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("The model hub did not return an immutable model revision")
    return commit


def _configure_determinism(enabled: bool) -> None:
    # Configure cuBLAS before the first CUDA operation. An explicit user setting
    # takes precedence and is recorded in environment.json.
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _load_pipeline(config: RunConfig, local_files_only: bool):
    import torch
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; activate a CUDA-enabled environment, or use "
            "--device cpu --dtype float32 --offload none"
        )
    if config.device == "cuda" and config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; use float16 or float32")
    if config.device == "cuda" and config.gpu_memory_limit_gib is not None:
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        fraction = config.gpu_memory_limit_gib / total_gib
        if fraction > 1:
            raise ValueError("Requested allocator limit exceeds the GPU capacity")
        torch.cuda.set_per_process_memory_fraction(float(fraction), device=0)
        print(f"PyTorch CUDA allocator capped at {config.gpu_memory_limit_gib:.2f} GiB", flush=True)
    dtype = getattr(torch, config.dtype)
    options = {
        "revision": config.revision,
        "torch_dtype": dtype,
        "use_safetensors": True,
        "local_files_only": local_files_only,
        "add_watermarker": False,
    }
    if config.variant is not None:
        options["variant"] = config.variant
    print(f"Loading {MODEL_ID} at {config.revision}", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **options)
    if config.scheduler_config is None:
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="leading",
            clip_sample=False,
            thresholding=False,
            set_alpha_to_one=True,
            steps_offset=0,
        )
    else:
        pipe.scheduler = DDIMScheduler.from_config(config.scheduler_config)
    if (
        pipe.scheduler.config.timestep_spacing != "leading"
        or pipe.scheduler.config.clip_sample
        or pipe.scheduler.config.thresholding
    ):
        raise ValueError("This backend requires leading DDIM timesteps without sample clipping or thresholding")
    if max(config.steps, config.tile_schedule_steps or config.steps) > pipe.scheduler.config.num_train_timesteps:
        raise ValueError("steps exceeds the model's training timestep count")
    if pipe.vae_scale_factor != 8 or pipe.unet.config.time_cond_proj_dim is not None:
        raise ValueError("The loaded architecture is not the expected SDXL base configuration")

    # Upcast before installing Accelerate hooks. Changing dtype after sequential
    # offload would leave its CPU weight map inconsistent with meta parameters.
    if pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast:
        pipe.vae.to(dtype=torch.float32)
    pipe.enable_vae_tiling()
    if config.attention_slicing:
        pipe.enable_attention_slicing()
    if config.device == "cpu":
        pipe.to("cpu")
    elif config.offload == "model":
        pipe.enable_model_cpu_offload(gpu_id=0)
    elif config.offload == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=0)
    else:
        pipe.to("cuda:0")
    return pipe


def _environment(config: RunConfig, pipe) -> dict[str, Any]:
    import torch

    versions = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "device": "cuda:0" if config.device == "cuda" else "cpu",
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allow_tf32": False,
        "attention_processors": sorted({type(p).__name__ for p in pipe.unet.attn_processors.values()}),
        "vae_dtype": str(pipe.vae.dtype),
        "vae_tiling": bool(pipe.vae.use_tiling),
        "gpu_memory_limit_gib": config.gpu_memory_limit_gib,
        "min_free_gpu_gib": config.min_free_gpu_gib,
    }
    if config.device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        info["gpu"] = {
            "name": properties.name,
            "total_memory_gib": properties.total_memory / 1024**3,
            "compute_capability": [properties.major, properties.minor],
        }
        info["sdp_backends"] = {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        }
    return info


def _decode_latent(pipe, latents, mode="standard"):
    import torch

    # This is always the official base VAE. Its precision was set before any
    # offload hooks were installed, so hooks and stored weights stay consistent.
    decode_latents = latents.to(dtype=pipe.vae.dtype)
    mean = getattr(pipe.vae.config, "latents_mean", None)
    std = getattr(pipe.vae.config, "latents_std", None)
    if mean is not None and std is not None:
        shape = (1, decode_latents.shape[1], 1, 1)
        mean = torch.tensor(mean, device=latents.device, dtype=decode_latents.dtype).view(shape)
        std = torch.tensor(std, device=latents.device, dtype=decode_latents.dtype).view(shape)
        decode_latents = decode_latents * std / pipe.vae.config.scaling_factor + mean
    else:
        decode_latents = decode_latents / pipe.vae.config.scaling_factor
    if mode == "cpu-tiles" and max(decode_latents.shape[-2:]) > pipe.vae.tile_latent_min_size:
        from .decode import decode_tiles_to_cpu

        decoded = decode_tiles_to_cpu(pipe.vae, decode_latents)
    else:
        decoded = pipe.vae.decode(decode_latents, return_dict=False)[0]
    if not torch.isfinite(decoded).all().item():
        raise RuntimeError("VAE decoding produced non-finite pixels; no invalid output was saved")
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    pipe.maybe_free_model_hooks()
    return image


def _synchronize(config: RunConfig) -> None:
    if config.device == "cuda":
        import torch

        torch.cuda.synchronize(0)


def _parent_target(config: RunConfig, pipe):
    """Crop a native tile from a completed, pinned PhotoQuilt parent generation."""
    from safetensors import safe_open

    source = config.parent_tile_source
    run = PROJECT_ROOT / source["run"]
    parent = RunConfig.read(run / "config.json")
    manifest = json.loads((run / "manifest.json").read_text())
    name = "final_latents.safetensors"
    if (manifest["status"] != "complete" or parent.model_id != config.model_id
            or parent.revision != config.revision or parent.dtype != config.dtype
            or manifest["outputs"][name]["sha256"] != source["sha256"]
            or _file_hash(run / name) != source["sha256"]):
        raise ValueError("Parent run or latent does not match the nested recipe")
    row, col = source["row"], source["column"]
    side = parent.tile_size // pipe.vae_scale_factor
    if (row >= parent.height // parent.tile_size or col >= parent.width // parent.tile_size
            or config.bootstrap_dimensions() != (parent.tile_size, parent.tile_size)):
        raise ValueError("Requested target is not a native tile of the parent")
    with safe_open(run / name, framework="pt", device="cpu") as handle:
        latent = handle.get_slice("latents")
        expected = [1, pipe.unet.config.in_channels, parent.height // pipe.vae_scale_factor,
                    parent.width // pipe.vae_scale_factor]
        if latent.get_shape() != expected:
            raise ValueError("Unexpected parent latent dimensions")
        return latent[:, :, row*side:(row+1)*side, col*side:(col+1)*side].contiguous().to(config.device)


def _generate(config: RunConfig, pipe, out_dir: Path, manifest: dict[str, Any]) -> None:
    import torch
    import torch.nn.functional as functional

    from .tiling import apply_tile_noise_seeds, tiled_tail_denoise

    device = torch.device("cuda:0" if config.device == "cuda" else "cpu")
    dtype = getattr(torch, config.dtype)
    boot_height, boot_width = config.bootstrap_dimensions()
    schedule_steps = config.tile_schedule_steps or config.steps
    manifest["derived"] = {
        "target_height": boot_height,
        "target_width": boot_width,
        "tile_rows": config.height // config.tile_size,
        "tile_columns": config.width // config.tile_size,
        "strength_step_fraction": config.noise_steps / schedule_steps,
        "ddim_eta": 0.0,
        "overlap_pixels": 0,
        "rng_device": "cpu",
        "scheduler_class": type(pipe.scheduler).__name__,
        "decode_mode": config.decode_mode,
        "tail_offload": config.tail_offload or config.offload,
        "tail_serial_cfg": config.tail_serial_cfg,
    }
    manifest["timings_seconds"] = {}
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats(0)

    with torch.inference_mode():
        _synchronize(config)
        started = time.perf_counter()
        if config.parent_tile_source is not None:
            target = _parent_target(config, pipe)
            print("Loaded the selected native tile from the verified parent latent", flush=True)
        else:
            print(f"Generating target {boot_width}x{boot_height}, seed={config.target_seed}", flush=True)
            target = pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                height=boot_height,
                width=boot_width,
                original_size=(boot_height, boot_width),
                target_size=(boot_height, boot_width),
                crops_coords_top_left=(0, 0),
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(config.target_seed),
                eta=0.0,
                output_type="latent",
            ).images
        _synchronize(config)
        manifest["timings_seconds"]["parent_tile_load" if config.parent_tile_source else "target_denoise"] = time.perf_counter() - started
        _decode_latent(pipe, target, config.decode_mode).save(out_dir / "target.png")
        manifest["samples_generated"]["targets"] = int(config.parent_tile_source is None)
        if config.parent_tile_source:
            manifest["derived"]["parent_tile_source"] = config.parent_tile_source
        manifest["outputs"]["target.png"] = {"sha256": _file_hash(out_dir / "target.png")}
        _write_json(out_dir / "manifest.json", manifest)

        print("Encoding tile text with both SDXL text encoders", flush=True)
        positive, negative, pooled, negative_pooled = pipe.encode_prompt(
            prompt=config.tile_prompt,
            negative_prompt=config.negative_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=config.guidance_scale > 1.0,
        )
        tile_dimensions = (config.tile_size, config.tile_size)
        time_ids = pipe._get_add_time_ids(
            original_size=tile_dimensions,
            crops_coords_top_left=(0, 0),
            target_size=tile_dimensions,
            dtype=positive.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(device)
        # Direct encode_prompt/decode calls are outside the normal pipeline
        # call order. Release their offload hooks before entering the UNet loop.
        pipe.maybe_free_model_hooks()

        if config.tail_offload is not None and config.tail_offload != config.offload:
            pipe.remove_all_hooks()
            if config.tail_offload == "model":
                pipe.enable_model_cpu_offload(gpu_id=0)
            else:
                pipe.enable_sequential_cpu_offload(gpu_id=0)
            print(f"Tile denoising offload: {config.tail_offload}", flush=True)

        clean = functional.interpolate(
            target.float(),
            size=(config.height // pipe.vae_scale_factor, config.width // pipe.vae_scale_factor),
            mode="bicubic",
            align_corners=False,
        ).to(device=device, dtype=dtype)
        del target
        pipe.scheduler.set_timesteps(schedule_steps, device=device)
        inject_index = schedule_steps - config.noise_steps
        injection_timestep = pipe.scheduler.timesteps[inject_index]
        noise = torch.randn(
            clean.shape,
            generator=torch.Generator(device="cpu").manual_seed(config.noise_seed),
            device="cpu",
            dtype=dtype,
        )
        if config.tile_noise_seeds:
            apply_tile_noise_seeds(noise, config.tile_size // pipe.vae_scale_factor, config.tile_noise_seeds)
        noise = noise.to(device)
        palette_target = None
        if config.tile_palette_guidance:
            latents = noise
            palette_target = clean
        else:
            latents = pipe.scheduler.add_noise(clean, noise, injection_timestep.reshape(1))
        del clean, noise
        manifest["derived"]["tile_initialization"] = "pure_noise" if config.tile_palette_guidance else "renoised_target"
        manifest["derived"]["tile_start_timestep"] = int(injection_timestep.item())
        manifest["derived"]["injection_index"] = None if config.tile_palette_guidance else inject_index
        manifest["derived"]["injection_timestep"] = None if config.tile_palette_guidance else int(injection_timestep.item())
        manifest["derived"]["full_timesteps"] = pipe.scheduler.timesteps.cpu().tolist()
        print(
            f"Denoising {config.width}x{config.height} canvas, tile={config.tile_size}, "
            f"batch={config.tile_batch}, noise seed={config.noise_seed}",
            flush=True,
        )
        _write_json(out_dir / "manifest.json", manifest)
        _synchronize(config)
        started = time.perf_counter()
        latents = tiled_tail_denoise(
            latents=latents,
            timesteps=pipe.scheduler.timesteps[inject_index:],
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            prompt_embeds=positive,
            negative_prompt_embeds=negative,
            pooled_prompt_embeds=pooled,
            negative_pooled_prompt_embeds=negative_pooled,
            add_time_ids=time_ids,
            negative_add_time_ids=time_ids if config.guidance_scale > 1.0 else None,
            guidance_scale=config.guidance_scale,
            tile_latent=config.tile_size // pipe.vae_scale_factor,
            tile_batch=config.tile_batch,
            serial_cfg=config.tail_serial_cfg,
            palette_target=palette_target,
            palette_strength=config.tile_palette_guidance,
        )
        _synchronize(config)
        manifest["timings_seconds"]["tile_denoise"] = time.perf_counter() - started
        # Preserve the expensive denoising result before decoding. This is an
        # output of our own pipeline, not an external image or model input.
        from safetensors.torch import save_file

        save_file({"latents": latents.detach().cpu().contiguous()}, str(out_dir / "final_latents.safetensors"))
        manifest["outputs"]["final_latents.safetensors"] = {"sha256": _file_hash(out_dir / "final_latents.safetensors")}
        _write_json(out_dir / "manifest.json", manifest)
        pipe.maybe_free_model_hooks()
        started = time.perf_counter()
        image = _decode_latent(pipe, latents, config.decode_mode)
        _synchronize(config)
        manifest["timings_seconds"]["mosaic_decode"] = time.perf_counter() - started
        image.save(out_dir / "mosaic.png")
        manifest["samples_generated"]["mosaics"] = 1
        manifest["outputs"]["mosaic.png"] = {"sha256": _file_hash(out_dir / "mosaic.png")}
        if config.save_tiles:
            tiles_dir = out_dir / "tiles"
            tiles_dir.mkdir()
            for row, top in enumerate(range(0, config.height, config.tile_size)):
                for column, left in enumerate(range(0, config.width, config.tile_size)):
                    tile_path = tiles_dir / f"r{row:03d}_c{column:03d}.png"
                    image.crop((left, top, left + config.tile_size, top + config.tile_size)).save(tile_path)
                    manifest["outputs"][str(tile_path.relative_to(out_dir))] = {"sha256": _file_hash(tile_path)}
        if config.device == "cuda":
            manifest["memory"] = {
                "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 1024**3,
            }


def _compare_replay(replay_path: Path, manifest: dict[str, Any], environment: dict[str, Any]) -> None:
    old_manifest_path = replay_path.parent / "manifest.json"
    if not old_manifest_path.is_file():
        return
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    comparisons = {}
    for name, record in manifest["outputs"].items():
        expected = old_manifest.get("outputs", {}).get(name, {}).get("sha256")
        if expected is not None:
            comparisons[name] = record["sha256"] == expected
    result = {
        "original_run": str(replay_path.parent),
        "output_sha256_matches": comparisons,
        "source_matches": manifest["source_sha256"] == old_manifest.get("source_sha256"),
    }
    old_environment_path = replay_path.parent / "environment.json"
    if old_environment_path.is_file():
        old_environment = json.loads(old_environment_path.read_text(encoding="utf-8"))
        result["package_versions_match"] = old_environment.get("packages") == environment["packages"]
    manifest["replay_comparison"] = result
    for name, matches in comparisons.items():
        print(f"Replay SHA256 {name}: {'MATCH' if matches else 'DIFFERENT'}", flush=True)


def main() -> None:
    config, output_root, local_files_only, replay_path = parse_options()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_dir = output_root / f"sdxl_mosaic_{run_id}_seed{config.seed}"
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "initializing",
        "started_at_utc": _utc_now(),
        "replay_config": str(replay_path) if replay_path is not None else None,
        "source_sha256": _snapshot_source(out_dir),
        "git": _git_provenance(),
        "samples_generated": {"targets": 0, "mosaics": 0},
        "outputs": {},
        "references": {
            "method": "https://arxiv.org/abs/2606.30968v2",
            "upstream_code": "https://github.com/KooroshRH/PhotoQuilt",
            "sdxl": "https://arxiv.org/abs/2307.01952",
            "model": f"https://huggingface.co/{MODEL_ID}",
            "ddim": "https://arxiv.org/abs/2010.02502",
            "diffusers": "https://github.com/huggingface/diffusers",
        },
    }
    _write_json(out_dir / "requested_config.json", config.to_dict())
    _write_json(out_dir / "manifest.json", manifest)
    pipe = None
    guard = GpuMemoryGuard(config.min_free_gpu_gib, out_dir)
    started = time.perf_counter()
    try:
        guard.__enter__()
        _configure_determinism(config.deterministic)
        config = replace(config, revision=_resolve_revision(config.revision, local_files_only))
        pipe = _load_pipeline(config, local_files_only)
        config = replace(config, scheduler_config=dict(pipe.scheduler.config))
        config.validate(replay=True)
        _write_json(out_dir / "config.json", config.to_dict())
        environment = _environment(config, pipe)
        _write_json(out_dir / "environment.json", environment)
        (out_dir / "requirements-runtime.txt").write_text(
            "# Observed runtime versions; install the appropriate PyTorch/CUDA build separately.\n"
            + "\n".join(f"{name}=={version}" for name, version in environment["packages"].items() if version is not None)
            + "\n",
            encoding="utf-8",
        )
        manifest["status"] = "generating"
        _write_json(out_dir / "manifest.json", manifest)
        _generate(config, pipe, out_dir, manifest)
        if replay_path is not None:
            _compare_replay(replay_path, manifest, environment)
        manifest["status"] = "complete"
        manifest["finished_at_utc"] = _utc_now()
        manifest["total_seconds"] = time.perf_counter() - started
        _write_json(out_dir / "manifest.json", manifest)
        print(f"Saved target, mosaic, and reproduction recipe to {out_dir}", flush=True)
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, MemoryGuardStop)) else "failed"
        manifest["finished_at_utc"] = _utc_now()
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(out_dir / "manifest.json", manifest)
        raise
    finally:
        guard.close()


if __name__ == "__main__":
    main()
