"""Generate five new SDXL PhotoQuilt boards as one cyclic diffusion trajectory."""
import argparse
import fcntl
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .config import RunConfig
from .hero_cycle import Tail, write_json
from .joint_zoom import ARMS, RECIPE, conditioning, guide_pair, lowpass, match_slot, resize, split_board
from .run_sdxl_mosaic import _configure_determinism, _decode_latent, _environment, _load_pipeline

HEROES = ("spider_man", "iron_man", "captain_america", "hulk", "thor")
PORTRAITS = (
    "Spider-Man in a crimson webbed mask, a single bold three-quarter close portrait, large ivory eyes, blue-black city shadows, cinematic light.",
    "Iron Man's iconic red and brushed gold helmet fills the frame, glowing eye slits, burgundy reflections, graphite steel workshop in shadow.",
    "Captain America in his navy A helmet, a large clear face and a curved red shield rim, warm copper light and navy shadows, cinematic portrait.",
    "The Hulk's unmistakable green face and broad furrowed brow filling the frame, moss green light and midnight blue industrial shadows.",
    "Thor with long blond hair and a short beard, a single large face, crimson cape, silver armor and a dark stormy teal sky, cinematic portrait.",
)
SUBJECTS = ("Spider-Man in his red webbed mask and blue suit", "Iron Man in red and gold armor",
            "Captain America wearing his navy A helmet and carrying his star shield", "the muscular green Hulk",
            "Thor with blond hair, a red cape and his hammer")
POSES = ("a cinematic head-and-shoulders portrait of", "a cinematic medium shot of", "a cinematic action portrait of")
GRID, TILE, STEPS, SEED = 16, 768, 12, 72460
COUNT = GRID * GRID
LIGHTS = ("inside a shadowed steel hall", "against deep crimson brick and copper reflections",
          "in warm amber light", "against midnight-blue steel", "in green industrial light",
          "in silver daylight with a layered background")


def scene_ids(target):
    values = target.resize((GRID, GRID), Image.Resampling.BOX).get_flattened_data()
    result = []
    for i, (r, g, b) in enumerate(values):
        if max(r, g, b) < 59:
            light = 0
        elif r > 1.22 * g:
            light = 1
        elif r > 1.12 * b and g > 1.12 * b:
            light = 2
        elif b > 1.12 * r:
            light = 3
        elif g > 1.12 * r:
            light = 4
        else:
            light = 5
        row, col = divmod(i, GRID)
        result.append(((row + 2 * col) % len(POSES), light))
    return result


def save_latents(path, tensor, slots=()):
    tmp = path.with_suffix(".tmp")
    save_file({"latents": tensor.cpu().contiguous(), "slots": torch.tensor(slots, dtype=torch.long)}, str(tmp))
    tmp.replace(path)


def make_targets(pipe, root, portraits=PORTRAITS, seed=SEED, tile_prompts=None, *, target_seeds=None):
    anchors, prompts = [], []
    for j, portrait in enumerate(portraits):
        folder = root / HEROES[j]
        folder.mkdir(exist_ok=True)
        path = folder / "target.safetensors"
        if path.exists():
            latent = load_file(str(path))["latents"].cuda()
        else:
            target_seed = seed + j if target_seeds is None else target_seeds[j]
            latent = pipe(portrait, negative_prompt="", height=1024, width=1024, num_inference_steps=24,
                          guidance_scale=5., eta=0., output_type="latent",
                          generator=torch.Generator().manual_seed(target_seed)).images
            save_latents(path, latent)
        image_path = folder / "target.png"
        if not image_path.exists():
            _decode_latent(pipe, latent).save(image_path)
        if tile_prompts is not None:
            prompts.append([tile_prompts[j - 1]] * COUNT)
            anchors.append(latent)
            continue
        with Image.open(image_path) as target:
            prompts.append([f"{POSES[pose]} {SUBJECTS[j-1]} {LIGHTS[light]}, directional light, detailed costume"
                            for pose, light in scene_ids(target)])
        anchors.append(latent)
    return anchors, prompts


@dataclass(frozen=True)
class TargetGuidance:
    gain: float = .23
    bands: int = 5
    last_step: int = 10  # One-based, inclusive.


def sample(pipe, anchors, prompts, guidance, root, *, reservation=False, seed=SEED, target_guidance=None):
    if reservation:
        from .dino_reservation import estimate_batch, guide_boundary, reserve, share_latents
    if target_guidance is None:
        target_guidance = [TargetGuidance() for _ in anchors]
    side = TILE // 8
    fields = torch.stack([split_board(resize(a.float(), side * GRID), GRID) for a in anchors])
    pipe.maybe_free_model_hooks()
    pipe.remove_all_hooks()
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    cond = {p: conditioning(pipe, p) for p in sorted({p for stage in prompts for p in stage})}
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.unet.to("cuda")
    tail = Tail(pipe.scheduler, STEPS)
    states = sorted(root.glob("step_??.safetensors"))
    if states:
        last = states[-1]
        start = int(last.stem.split("_")[1])
        data = load_file(str(last))
        z, slots = data["latents"].cuda(), data["slots"].tolist()
    else:
        start, slots = 0, []
        noise = torch.randn(fields.shape, dtype=anchors[0].dtype, generator=torch.Generator().manual_seed(seed + 5)).cuda()
        z = torch.stack([pipe.scheduler.add_noise((.85 * lowpass(f, 12)).half(), n, tail.times[:1])
                         for f, n in zip(fields, noise)])
    for k in range(start, STEPS):
        t = tail.times[k]
        clean, following = torch.empty_like(z), torch.empty_like(z)
        for j, stage_prompts in enumerate(prompts):
            if reservation:
                estimate_batch(pipe, tail, z[j], t, cond[stage_prompts[0]], clean[j], following[j],
                               slots[j] if slots else None)
                print(f"step {k+1}/{STEPS}: {HEROES[j]}", flush=True)
                continue
            for i, p in enumerate(stage_prompts):
                tile = z[j, i:i + 1]
                u, v = pipe.unet(tile.repeat(2, 1, 1, 1), t, **cond[p], return_dict=False)[0].chunk(2)
                following[j, i:i + 1], clean[j, i:i + 1] = tail.step(u + 5.5 * (v-u), t, tile)
            print(f"step {k+1}/{STEPS}: {HEROES[j]}", flush=True)
        alpha = float(tail.alphas[int(t)][1]) ** .5
        if k == 3:
            if reservation:
                ranking = reserve(pipe, clean, GRID)
                slots = [item["slot"] for item in ranking]
                write_json(root / "reservation.json", ranking)
            else:
                slots = [match_slot(clean[j], clean[j-1], GRID) for j in range(len(HEROES))]
        if reservation and slots:
            share_latents(clean, slots, GRID)
        for j, guide in enumerate(target_guidance):
            if 4 <= k + 1 <= guide.last_step:
                following[j].add_(alpha * guide.gain * lowpass(fields[j] - clean[j].float(), guide.bands))
        if reservation and slots:
            if k <= 10:
                for j, slot in enumerate(slots):
                    guide_boundary(following[j], clean[j], clean[j-1], slot, GRID, alpha * .25)
            share_latents(following, slots, GRID)
        elif 3 <= k <= guidance.last_step:
            for j, slot in enumerate(slots):
                guide_pair(following[j], following[j-1], clean[j], clean[j-1], GRID, slot,
                           alpha * guidance.strength, guidance.bands)
        z = following
        path = root / f"step_{k + 1:02}.safetensors"
        save_latents(path, z, slots)
        if k:
            (root / f"step_{k:02}.safetensors").unlink(missing_ok=True)
    pipe.unet.to("cpu")
    pipe.vae.to("cuda")
    return z, slots


@torch.inference_mode()
def generate(root, arm):
    _configure_determinism(True)
    cfg = replace(RunConfig.read(RECIPE), offload="model")
    recipe = dict(heroes=HEROES, grid=GRID, tile_size=TILE, seed=SEED, target_prompts=PORTRAITS,
                  model=cfg.model_id, revision=cfg.revision, profile=arm, guidance=asdict(ARMS[arm]),
                  tile_steps=STEPS, target_steps=24, noise_seed=SEED+5, target_seeds=list(range(SEED, SEED+5)))
    path = root / "recipe.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(recipe)):
        raise ValueError("Output directory belongs to a different recipe.")
    write_json(path, recipe)
    pipe = _load_pipeline(cfg, True)
    anchors, prompts = make_targets(pipe, root)
    write_json(root / "prompts.json", prompts)
    z, slots = sample(pipe, anchors, prompts, ARMS[arm], root)
    for j, hero in enumerate(HEROES):
        folder = root / hero
        png = folder / "board.png"
        if not png.exists():
            canvas = Image.new("RGB", (GRID*TILE,) * 2)
            for i in range(COUNT):
                tile = _decode_latent(pipe, z[j, i:i + 1])
                row, col = divmod(i, GRID)
                canvas.paste(tile, (col*TILE, row*TILE))
                if i == slots[j]:
                    tile.save(folder / "portal.png")
            canvas.resize((1024, 1024), Image.Resampling.BOX).save(folder / "overview.png")
            tmp = png.with_suffix(".tmp.png")
            canvas.save(tmp, compress_level=1)
            tmp.replace(png)
            del canvas
        print(f"decoded {hero}", flush=True)
    write_json(root / "generation.json", dict(slots=slots, environment=_environment(cfg, pipe),
               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, tile_count=len(HEROES)*COUNT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generate(args.output, args.arm)
