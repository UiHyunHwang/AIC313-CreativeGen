"""Decode newly sampled leaves and render a five-level cycle."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from PIL import Image
from safetensors.torch import load_file, save_file
import torch

from .config import RunConfig
from .hero_cycle import write_json
from .joint_cycle import GRID, HEROES, RECIPE, TILE
from .render_joint_cycle import render
from .run_sdxl_mosaic import _configure_determinism, _decode_latent, _load_pipeline


def origin(slot, size, grid=GRID):
    row, col = divmod(slot, grid)
    return col*size, row*size


def resolve_views(templates, slots, grid=GRID):
    """Rasterize cyclic board references down to the last resolvable pixel."""
    views = templates
    for _ in range(16):
        following = [im.copy() for im in templates]
        for j, slot in enumerate(slots):
            size = templates[j].width // grid
            following[j].paste(views[j-1].reduce(grid), origin(slot, size, grid))
        if all(a.tobytes() == b.tobytes() for a, b in zip(views, following)):
            return following
        views = following
    raise RuntimeError("Cyclic view rasters did not converge")


def comparison(root):
    """Top row: departing view. Bottom row: the actual tile reached at the cut."""
    side = 256
    out = Image.new("RGB", (len(HEROES) * side, 2 * side))
    for j, hero in enumerate(HEROES):
        parent = HEROES[(j + 1) % len(HEROES)]
        paths = (root / hero / "overview.png", root / parent / "portal.png")
        for r, path in enumerate(paths):
            with Image.open(path) as image:
                out.paste(image.resize((side, side), Image.Resampling.LANCZOS), (j * side, r * side))
    out.save(root / "cut_comparison.jpg", quality=94)


@torch.inference_mode()
def vae_views(pipe, templates, slots, root):
    """Solve five reserved SDXL VAE latents against the current RGB board previews."""
    views = templates
    for _ in range(4):
        following = [im.copy() for im in templates]
        for j, slot in enumerate(slots):
            pixels = pipe.image_processor.preprocess(views[j-1]).to("cuda", dtype=pipe.vae.dtype)
            latent = pipe.vae.encode(pixels).latent_dist.mode() * pipe.vae.config.scaling_factor
            portal = _decode_latent(pipe, latent)
            folder = root / HEROES[j]
            portal.save(folder / "portal.png")
            save_file({"latent": latent.cpu().contiguous()}, str(folder / "portal.safetensors"))
            following[j].paste(portal.reduce(GRID), origin(slot, TILE // GRID))
        views = following
    return views


@torch.inference_mode()
def export(source, root, shared):
    _configure_determinism(True)
    data = json.loads((source / "generation.json").read_text())
    slots = data["slots"]
    root.mkdir(exist_ok=True, parents=True)
    write_json(root / "generation.json", data)
    z = load_file(str(source / "step_12.safetensors"))["latents"]
    pipe = _load_pipeline(replace(RunConfig.read(RECIPE), offload="model"), True)
    pipe.maybe_free_model_hooks()
    pipe.remove_all_hooks()
    pipe.unet = pipe.text_encoder = pipe.text_encoder_2 = None
    pipe.vae.to("cuda")
    templates = []
    Image.MAX_IMAGE_PIXELS = (TILE * GRID)**2
    for j, hero in enumerate(HEROES):
        folder = root / hero
        folder.mkdir(exist_ok=True)
        im = Image.new("RGB", (TILE*GRID,)*2)
        with Image.open(source / hero / "target.png") as target:
            target.save(folder / "target.png")
        for i, latent in enumerate(z[j]):
            if i == slots[j]:
                continue  # This cell references a board; it has no independent RGB leaf.
            tile = _decode_latent(pipe, latent[None].cuda())
            im.paste(tile, origin(i, TILE))
        templates.append(im.reduce(GRID))
        im.save(folder / "leaves.png", compress_level=1)
        print("decoded", hero, flush=True)
        del im
    views = resolve_views(templates, slots) if shared else vae_views(pipe, templates, slots, root)
    for j, hero in enumerate(HEROES):
        folder = root / hero
        views[j].save(folder / "overview.png")
        if shared:
            views[j-1].save(folder / "portal.png")
        leaves = folder / "leaves.png"
        with Image.open(leaves) as canvas, Image.open(folder / "portal.png") as portal:
            canvas.load()
            canvas.paste(portal, origin(slots[j], TILE))
            if canvas.reduce(GRID).tobytes() != views[j].tobytes():
                raise ValueError("Native board differs from its declared overview")
            canvas.save(folder / "board.png", compress_level=1)
        leaves.unlink()
    render(root)
    comparison(root)
    write_json(root / "view_mode.json", dict(mode="shared-rgb" if shared else "shared-latent",
               exact_child_pixels=shared))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--mode", choices=("shared-latent", "shared-rgb"), required=True)
    args = parser.parse_args()
    export(args.source, args.source / args.mode, args.mode == "shared-rgb")
