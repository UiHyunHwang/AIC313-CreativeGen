"""Serializable generation recipe; contains no image paths or extra model IDs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MAX_SEED = 2**63 - 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunConfig:
    prompt: str
    tile_prompt: str | None = None
    negative_prompt: str = ""
    seed: int = 0
    target_seed: int | None = None
    noise_seed: int | None = None
    tile_noise_seeds: dict[str, int] | None = None
    parent_tile_source: dict[str, Any] | None = None
    height: int = 4096
    width: int = 4096
    bootstrap_size: int = 1024
    tile_size: int = 1024
    steps: int = 50
    noise_steps: int = 30
    tile_schedule_steps: int | None = None
    tile_palette_guidance: float = 0.0
    guidance_scale: float = 5.0
    tile_batch: int = 1
    dtype: str = "float16"
    device: str = "cuda"
    offload: str = "model"
    tail_offload: str | None = None
    tail_serial_cfg: bool = False
    decode_mode: str = "standard"
    attention_slicing: bool = False
    deterministic: bool = True
    save_tiles: bool = False
    gpu_memory_limit_gib: float | None = None
    min_free_gpu_gib: float = 0.0
    model_id: str = MODEL_ID
    revision: str | None = None
    variant: str | None = None
    scheduler_config: dict[str, Any] | None = None
    schema_version: int = 1

    def normalized(self) -> "RunConfig":
        return replace(
            self,
            tile_prompt=self.prompt if self.tile_prompt is None else self.tile_prompt,
            target_seed=self.seed if self.target_seed is None else self.target_seed,
            noise_seed=(self.seed + 1) % (MAX_SEED + 1) if self.noise_seed is None else self.noise_seed,
            variant="fp16" if self.dtype == "float16" else None,
        )

    def validate(self, *, replay: bool = False) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported config schema_version")
        if self.model_id != MODEL_ID:
            raise ValueError(f"This backend only supports the allowed checkpoint {MODEL_ID}")
        for name in ("prompt", "tile_prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty team-authored text prompt")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("negative_prompt must be a string")
        for name in ("seed", "target_seed", "noise_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= MAX_SEED:
                raise ValueError(f"{name} must be an integer between 0 and {MAX_SEED}")
        for name in ("height", "width", "bootstrap_size", "tile_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 8 or value % 8:
                raise ValueError(f"{name} must be a positive multiple of 8")
        if self.height % self.tile_size or self.width % self.tile_size:
            raise ValueError("height and width must be exact multiples of tile_size (no overlapping tiles)")
        if self.parent_tile_source is not None:
            source = self.parent_tile_source
            if not isinstance(source, dict) or set(source) != {"run", "sha256", "row", "column"}:
                raise ValueError("parent_tile_source requires run, sha256, row and column")
            if not isinstance(source["run"], str) or not source["run"].strip():
                raise ValueError("Missing parent run directory")
            if not isinstance(source["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"]):
                raise ValueError("Expected a SHA-256 for the parent final latent")
            if any(type(source[key]) is not int or source[key] < 0 for key in ("row", "column")):
                raise ValueError("Parent tile coordinates must be nonnegative integers")
        if self.tile_noise_seeds is not None:
            if not isinstance(self.tile_noise_seeds, dict):
                raise ValueError("tile_noise_seeds must map tile identifiers to seeds")
            for key, seed in self.tile_noise_seeds.items():
                match = re.fullmatch(r"r([0-9]{3,})_c([0-9]{3,})", key) if isinstance(key, str) else None
                if match is None:
                    raise ValueError(f"Invalid tile identifier: {key!r}")
                row, column = map(int, match.groups())
                if key != f"r{row:03d}_c{column:03d}" or row >= self.height // self.tile_size or column >= self.width // self.tile_size:
                    raise ValueError(f"Invalid tile coordinate: {key!r}")
                if type(seed) is not int or not 0 <= seed <= MAX_SEED:
                    raise ValueError(f"Invalid seed for {key!r}")
        for name in ("steps", "noise_steps", "tile_batch"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.steps > 1000:
            raise ValueError("Bootstrap schedule exceeds 1000 training timesteps")
        if self.tile_schedule_steps is None:
            if self.noise_steps >= self.steps:
                raise ValueError("Require noise_steps < steps with the default tile schedule")
        elif type(self.tile_schedule_steps) is not int or not self.noise_steps <= self.tile_schedule_steps <= 1000:
            raise ValueError("Require noise_steps <= tile_schedule_steps <= 1000")
        if type(self.tile_palette_guidance) not in (int, float) or not 0 <= self.tile_palette_guidance <= 1:
            raise ValueError("tile_palette_guidance must be between 0 and 1")
        if self.tile_palette_guidance and self.tile_schedule_steps != self.noise_steps:
            raise ValueError("Palette guidance uses a full tile schedule starting from pure noise")

        if (
            isinstance(self.guidance_scale, bool)
            or not isinstance(self.guidance_scale, (int, float))
            or not math.isfinite(self.guidance_scale)
            or self.guidance_scale < 0
        ):
            raise ValueError("guidance_scale must be finite and non-negative")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda (one GPU) or cpu")
        if self.offload not in ("none", "model", "sequential"):
            raise ValueError("offload must be none, model, or sequential")
        if self.tail_offload not in (None, "model", "sequential"):
            raise ValueError("tail_offload must be null, model, or sequential")
        if self.device == "cpu" and self.tail_offload is not None:
            raise ValueError("tail_offload requires CUDA")
        if self.decode_mode not in ("standard", "cpu-tiles"):
            raise ValueError("decode_mode must be standard or cpu-tiles")
        if self.device == "cpu" and (self.dtype != "float32" or self.offload != "none"):
            raise ValueError("CPU execution requires --dtype float32 --offload none")
        for name in ("gpu_memory_limit_gib", "min_free_gpu_gib"):
            value = getattr(self, name)
            if value is None and name == "gpu_memory_limit_gib":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.gpu_memory_limit_gib == 0:
            raise ValueError("gpu_memory_limit_gib must be positive or null")
        if self.device == "cpu" and (self.gpu_memory_limit_gib is not None or self.min_free_gpu_gib):
            raise ValueError("GPU memory limits require --device cuda")
        for name in ("attention_slicing", "deterministic", "save_tiles", "tail_serial_cfg"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.variant != ("fp16" if self.dtype == "float16" else None):
            raise ValueError("variant must be fp16 for float16, and null otherwise")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("revision must be a non-empty Hugging Face revision")
        if self.scheduler_config is not None and not isinstance(self.scheduler_config, dict):
            raise ValueError("scheduler_config must be an object")
        if replay:
            if self.revision is None or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
                raise ValueError("Replay requires an immutable 40-character model revision in config.json")
            if self.scheduler_config is None:
                raise ValueError("Replay requires the scheduler_config saved by a previous run")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def read(cls, path: Path) -> "RunConfig":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("config.json must contain a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
        config = cls(**values)
        # A saved recipe is already normalized. Do not silently change its seeds,
        # precision, prompts, or model variant when reproducing it.
        config.validate(replay=True)
        return config

    def bootstrap_dimensions(self) -> tuple[int, int]:
        scale = min(1.0, self.bootstrap_size / max(self.height, self.width))
        height = max(8, int(round(self.height * scale)) // 8 * 8)
        width = max(8, int(round(self.width * scale)) // 8 * 8)
        return height, width


def parse_options() -> tuple[RunConfig, Path, bool, Path | None]:
    parser = argparse.ArgumentParser(
        description="Generate a target and PhotoQuilt mosaic entirely with the allowed SDXL base.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=Path, help="Replay a complete saved config.json")
    parser.add_argument("--output-dir", type=Path, help="Parent directory for a new run folder")
    parser.add_argument("--local-files-only", action="store_true", help="Use only already cached official weights")
    parser.add_argument("--prompt", help="Team-authored global target prompt (required for a new run)")
    parser.add_argument("--tile-prompt", help="Team-authored tile prompt; defaults to --prompt")
    parser.add_argument("--negative-prompt", help="Negative text used for both passes; defaults to an empty string")
    for flag in ("seed", "target-seed", "noise-seed", "height", "width", "bootstrap-size", "tile-size", "steps", "noise-steps", "tile-schedule-steps", "tile-batch"):
        parser.add_argument(f"--{flag}", type=int)
    parser.add_argument("--tile-noise-seeds", type=json.loads, help="JSON mapping of tile IDs to independent noise seeds")
    parser.add_argument("--tile-palette-guidance", type=float, help="Strength of 2x2 latent palette guidance for pure-noise tiles")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--offload", choices=("none", "model", "sequential"))
    parser.add_argument("--tail-offload", choices=("model", "sequential"))
    parser.add_argument("--tail-serial-cfg", action="store_true", help="Evaluate negative/positive tile conditions separately to reduce activation memory")
    parser.add_argument("--decode-mode", choices=("standard", "cpu-tiles"))
    parser.add_argument("--attention-slicing", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-tiles", action="store_true", help="Also save unedited crops of the final mosaic")
    parser.add_argument("--gpu-memory-limit-gib", type=float, help="Limit this process's PyTorch CUDA allocator")
    parser.add_argument("--min-free-gpu-gib", type=float, help="Stop only this generation if GPU free memory falls below this threshold")
    parser.add_argument("--revision", help="Official model revision; resolved and recorded as a commit SHA")
    values = vars(parser.parse_args())
    replay_path = values.pop("config", None)
    output_dir = values.pop("output_dir", PROJECT_ROOT / "outputs" / "sdxl")
    local_files_only = values.pop("local_files_only", False)
    try:
        if replay_path is not None:
            if values:
                parser.error("--config cannot be combined with generation flags; use its saved recipe unchanged")
            replay_path = replay_path.expanduser().resolve()
            config = RunConfig.read(replay_path)
        else:
            if "prompt" not in values:
                parser.error("--prompt is required unless --config is supplied")
            config = RunConfig(**values).normalized()
            config.validate()
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    return config, output_dir.expanduser().resolve(), local_files_only, replay_path
