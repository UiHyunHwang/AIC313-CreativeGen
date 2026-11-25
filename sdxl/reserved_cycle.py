"""Generate the leaf imagery of a cyclic mosaic with five pre-reserved child views."""
import argparse
import fcntl
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .config import RunConfig
from .dino_reservation import BATCH, MODEL, REVISION, WEIGHTS
from .hero_cycle import write_json
from .joint_cycle import GRID, HEROES, RECIPE, STEPS, TILE, TargetGuidance, make_targets, sample
from .run_sdxl_mosaic import _configure_determinism, _environment, _load_pipeline

PROMPTS = Path(__file__).with_name("reserved_cycle.json")
@torch.inference_mode()
def generate(root):
    _configure_determinism(True)
    cfg = replace(RunConfig.read(RECIPE), offload="model")
    authored = json.loads(PROMPTS.read_text())
    rows, seed = authored["heroes"], authored["seed"]
    if [row["hero"] for row in rows] != list(HEROES):
        raise ValueError("Hero order does not match the camera path")
    target_guidance = [TargetGuidance(**row.get("target_guidance", {})) for row in rows]
    target_seeds = [row.get("target_seed", seed + j) for j, row in enumerate(rows)]
    recipe = dict(**authored, model=cfg.model_id, revision=cfg.revision, dino=MODEL, dino_revision=REVISION,
                  weights=WEIGHTS, grid=GRID, tile_size=TILE, tile_steps=STEPS, reservation_step=4,
                  batch=BATCH, target_steps=24, target_seeds=target_seeds, noise_seed=seed+5)
    recipe["heroes"] = [dict(row, target_guidance=asdict(guide)) for row, guide in zip(rows, target_guidance)]
    path = root / "recipe.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for row in previous["heroes"]:
            row["target_guidance"] = asdict(TargetGuidance(**row.get("target_guidance", {})))
        if previous != recipe:
            raise ValueError("Use a fresh folder for a different recipe")
    write_json(path, recipe)
    pipe = _load_pipeline(cfg, True)
    anchors, prompts = make_targets(pipe, root, [row["target"] for row in rows], seed, [row["tile"] for row in rows],
                                    target_seeds=target_seeds)
    _, slots = sample(pipe, anchors, prompts, None, root, reservation=True, seed=seed, target_guidance=target_guidance)
    write_json(root / "generation.json", dict(slots=slots, target_count=5, preview_candidates=5*GRID**2,
               final_ordinary_tiles=5*(GRID**2-1), reserved_cells=5, environment=_environment(cfg, pipe),
               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(exist_ok=True, parents=True)
    with (args.root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generate(args.root)
