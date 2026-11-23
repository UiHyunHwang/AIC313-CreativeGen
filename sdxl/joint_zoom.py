"""Sample two PhotoQuilt scales with a shared low-frequency DDIM constraint."""
import argparse
import fcntl
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import RunConfig
from .hero_cycle import Tail, write_json
from .run_sdxl_mosaic import _configure_determinism, _decode_latent, _environment, _load_pipeline

RECIPE = Path(__file__).with_name("hero_cycle_recipe.json")
TARGETS = (
    "A close portrait of the Hulk, green face and furrowed brow filling the frame, deep blue industrial shadows, cinematic light.",
    "Captain America wearing his navy A helmet, a clear face above his crimson shield in a dark steel workshop, cinematic portrait.",
)
TILES = (
    "Captain America in his blue helmet with his shield in a shadowed industrial workshop",
    "Iron Man in red and gold armor in a shadowed industrial workshop",
)
SIDE = 128  # SDXL base latent side at its native 1024 resolution


@dataclass(frozen=True)
class Guidance:
    strength: float
    last_step: int  # zero-based, inclusive
    bands: int


ARMS = {
    "control": Guidance(0., 0, 32),
    "joint": Guidance(.78, 10, 32),
    "gentle": Guidance(.35, 7, 8),
}


def resize(x, side):
    return F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)


def lowpass(x, side):
    return resize(F.adaptive_avg_pool2d(x.float(), side), x.shape[-1])


def split_board(x, grid):
    _, channels, height, width = x.shape
    return (x.reshape(channels, grid, height // grid, grid, width // grid)
            .permute(1, 3, 0, 2, 4).reshape(grid * grid, channels, height // grid, width // grid))


def board(tiles, grid):
    _, channels, height, width = tiles.shape
    return (tiles.reshape(grid, grid, channels, height, width).permute(2, 0, 3, 1, 4)
            .reshape(1, channels, grid * height, grid * width))


def predict_slot(clean, grid):
    """Choose the nearest coarse latent field before cross-scale guidance begins."""
    count = grid * grid
    overview = F.adaptive_avg_pool2d(board(clean[count:], grid).float(), 8)
    candidates = F.adaptive_avg_pool2d(clean[:count].float(), 8)
    costs = (candidates - overview).square().mean(dim=(1, 2, 3))
    return int(costs.argmin())


def couple(next_z, clean, grid, slot, gain, bands=32):
    """Modify the two next DDIM states in place, using simultaneous x0 estimates."""
    count = grid * grid
    child = board(clean[count:], grid)
    side = clean.shape[-1]
    small = F.avg_pool2d(child.float(), grid)
    residual = lowpass(small - clean[slot:slot + 1].float(), min(bands, side))
    next_z[slot:slot + 1].add_(gain * .85 * residual)
    next_z[count:].sub_(gain * .15 * split_board(resize(residual, grid * side), grid))


def conditioning(pipe, prompt):
    pos, neg, pooled, neg_pooled = pipe.encode_prompt(prompt=prompt, negative_prompt="", device="cuda")
    ids = pipe._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), dtype=pos.dtype,
                                 text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).cuda()
    return dict(encoder_hidden_states=torch.cat((neg, pos)),
                added_cond_kwargs=dict(text_embeds=torch.cat((neg_pooled, pooled)), time_ids=ids.repeat(2, 1)))


def denoise(pipe, tail, anchors, conditions, grid, slot, noise, guidance):
    count = grid * grid
    fields = torch.cat([split_board(resize(a.float(), SIDE * grid), grid) for a in anchors])
    initial = (.85 * lowpass(fields, 12)).to(noise.dtype)
    z = pipe.scheduler.add_noise(initial, noise, tail.times[:1])
    for k, t in enumerate(tail.times):
        clean = torch.empty_like(z)
        next_z = torch.empty_like(z)
        for i, tile in enumerate(z.split(1)):
            uncond, cond = pipe.unet(tile.repeat(2, 1, 1, 1), t, **conditions[i // count],
                                     return_dict=False)[0].chunk(2)
            next_z[i:i+1], clean[i:i+1] = tail.step(uncond + 5.5 * (cond - uncond), t, tile)
        alpha = float(tail.alphas[int(t)][1]) ** .5
        if k == 3 and slot is None:
            slot = predict_slot(clean, grid)
        if 3 <= k <= 9:
            next_z.add_(alpha * .23 * lowpass(fields - clean.float(), 5))
        if 3 <= k <= guidance.last_step:
            couple(next_z, clean, grid, slot, alpha * guidance.strength, guidance.bands)
        z = next_z
    return z, slot


def error(a, b, side):
    def pixels(im):
        return np.asarray(im.resize((side, side), Image.Resampling.BOX), dtype=np.float32) / 255
    return float(np.abs(pixels(a) - pixels(b)).mean())


def export(pipe, z, grid, slot, folder, targets):
    folder.mkdir(exist_ok=True)
    tile_side = 1024
    boards = []
    for level, name in enumerate(("parent", "child")):
        image = Image.new("RGB", (tile_side * grid,) * 2)
        for i in range(grid * grid):
            j = level * grid * grid + i
            tile = _decode_latent(pipe, z[j:j + 1])
            row, col = divmod(i, grid)
            image.paste(tile, (col * tile_side, row * tile_side))
            if level == 0 and i == slot:
                portal = tile
                portal.save(folder / "portal.png")
        image.save(folder / f"{name}.png", compress_level=2)
        boards.append(image)
    return dict(portal_child_mae_8=error(portal, boards[1], 8),
                portal_child_mae_32=error(portal, boards[1], 32),
                parent_hulk_mae_16=error(boards[0], targets[0], 16),
                child_captain_mae_16=error(boards[1], targets[1], 16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, choices=(2, 4, 8, 16), default=4)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--slot-policy", choices=("diffusion", "seed"), default="diffusion",
                        help="Reserve a seeded slot or infer it from the fourth SDXL clean prediction.")
    parser.add_argument("--arm", choices=("both", *ARMS), default="both",
                        help="Run one arm at a time when GPU session length is limited.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


@torch.inference_mode()
def run(args):
    _configure_determinism(True)
    cfg = replace(RunConfig.read(RECIPE), offload="model")
    pipe = _load_pipeline(cfg, True)
    count = args.grid ** 2
    initial_slot = random.Random(args.seed).randrange(count) if args.slot_policy == "seed" else None
    seeds = [args.seed + i for i in (1, 2, 3)]
    report = dict(model=cfg.model_id, revision=cfg.revision, grid=args.grid, seed=args.seed,
                  slot_policy=args.slot_policy, target_prompts=list(TARGETS), tile_prompts=list(TILES),
                  target_seeds=seeds[:2], shared_tile_noise_seed=seeds[2], target_steps=24, tile_steps=12,
                  tiles_per_arm=2*count, environment=_environment(cfg, pipe), arms={})
    manifest = args.output / "run.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        for key in ("model", "revision", "grid", "seed", "target_prompts", "tile_prompts", "slot_policy"):
            if previous.get(key, "seed" if key == "slot_policy" else None) != report[key]:
                raise ValueError(f"Existing run differs in {key}; use a fresh output directory")
        report.update(previous)
    write_json(manifest, report)
    anchors, targets = [], []
    for i, prompt in enumerate(TARGETS):
        latent = pipe(prompt, negative_prompt="", height=1024, width=1024, num_inference_steps=24,
                      guidance_scale=5., eta=0., output_type="latent",
                      generator=torch.Generator().manual_seed(seeds[i])).images
        anchors.append(latent)
        image = _decode_latent(pipe, latent)
        target_path = args.output / f"target_{i}.png"
        if target_path.exists():
            with Image.open(target_path) as saved:
                if saved.tobytes() != image.tobytes():
                    raise RuntimeError(f"Regenerated target differs from {target_path}")
        else:
            image.save(target_path)
        targets.append(image)
    pipe.maybe_free_model_hooks()
    pipe.remove_all_hooks()
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    conditions = [conditioning(pipe, prompt) for prompt in TILES]
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.unet.to("cuda")
    pipe.vae.to("cuda")
    tail = Tail(pipe.scheduler, 12)
    noise = torch.randn((2 * count, 4, SIDE, SIDE), dtype=anchors[0].dtype,
                        generator=torch.Generator().manual_seed(seeds[2])).cuda()
    names = ("control", "joint") if args.arm == "both" else (args.arm,)
    for name in names:
        guidance = ARMS[name]
        if name in report["arms"]:
            continue
        print(f"Sampling {name} with {count} tiles per level", flush=True)
        z, slot = denoise(pipe, tail, anchors, conditions, args.grid, initial_slot, noise, guidance)
        position = list(divmod(slot, args.grid))
        if report.setdefault("slot_zero_based", position) != position:
            raise RuntimeError("Matched arms resolved different portal coordinates")
        metrics = export(pipe, z, args.grid, slot, args.output / name, targets)
        report["arms"][name] = dict(strength=guidance.strength, last_step=guidance.last_step, bands=guidance.bands, **metrics)
        report["peak_reserved_gib"] = max(report.get("peak_reserved_gib", 0), torch.cuda.max_memory_reserved() / 2**30)
        write_json(manifest, report)
        print(metrics, flush=True)


if __name__ == "__main__":
    main()
