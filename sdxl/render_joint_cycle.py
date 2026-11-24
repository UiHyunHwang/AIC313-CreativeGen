"""Render only crops of the five jointly sampled boards; verify the encoded loop."""
import argparse
import json
from pathlib import Path

from PIL import Image

from .hero_cycle import write_json
from .hero_cycle_render import digest
from .joint_cycle import GRID, HEROES, TILE
from .joint_zoom import error

SIDE, STOP, FRAMES = 704, .90, 90
RESAMPLE = Image.Resampling.LANCZOS


def crop(image, left, top, width, side=SIDE):
    n = image.width
    x = max(0., min(left, 1. - width)) * n
    y = max(0., min(top, 1. - width)) * n
    return image.resize((side, side), RESAMPLE, box=(x, y, min(n, x + width*n), min(n, y + width*n)))


def frame(inner, levels, slot, k):
    t = k / FRAMES
    width = STOP * GRID ** (t*t*(3-2*t))
    if width <= 1:
        return crop(inner, (1-width)/2, (1-width)/2, width)
    width /= GRID
    mix = (width - 1/GRID) / (STOP - 1/GRID)
    row, col = divmod(slot, GRID)
    left = (1-mix)*col/GRID + mix*(1-STOP)/2
    top = (1-mix)*row/GRID + mix*(1-STOP)/2
    # Choose the smallest source still at least as detailed as the display.
    source = next((im for im in reversed(levels) if im.width * width >= SIDE), levels[0])
    return crop(source, left, top, width)


def render(root):
    data = json.loads((root / "generation.json").read_text())
    Image.MAX_IMAGE_PIXELS = (GRID*TILE)**2
    slots = data["slots"]
    overviews = []
    swatches = Image.new("RGB", (192*len(HEROES), 2*192))
    for j, hero in enumerate(HEROES):
        with Image.open(root / hero / "overview.png") as im:
            overviews.append(im.convert("RGB"))
            swatches.paste(im.resize((192,192), RESAMPLE), (j*192, 0))
        with Image.open(root / hero / "portal.png") as p:
            swatches.paste(p.resize((192,192), RESAMPLE), (j*192,192))
    palette = swatches.quantize(256, method=Image.Quantize.MEDIANCUT)

    def quantize(im):
        return im.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)

    frames, durations, edges = [], [], []
    palette_errors = {}
    sheet = Image.new("RGB", (5*256, 256))
    for j, hero in enumerate(HEROES):
        sheet.paste(overviews[j].resize((256,256), RESAMPLE), (j*256,0))
        outer = (j+1) % len(HEROES)
        folder = root / HEROES[outer]
        slot = slots[outer]
        row, col = divmod(slot, GRID)
        with Image.open(folder / "board.png") as native, Image.open(folder / "portal.png") as portal:
            if native.size != (GRID*TILE,)*2:
                raise ValueError("Incorrect native board size")
            tile = native.crop((col*TILE,row*TILE,(col+1)*TILE,(row+1)*TILE))
            if tile.tobytes() != portal.tobytes():
                raise ValueError("Encoded parent crop differs from generated portal")
            levels = [native] + [native.reduce(n) for n in (2,4,8,16)]
            edges.append(dict(inner=hero, outer=HEROES[outer], row=row+1, column=col+1,
                              cut_mae_32=error(portal, overviews[j], 32),
                              cut_mae_128=error(portal, overviews[j], 128)))
            hold = frame(overviews[j], levels, slot, 0)
            encoded = quantize(hold)
            frames.append(encoded)
            palette_errors[hero] = error(hold, encoded.convert("RGB"), 128)
            durations.append(5500)
            for k in range(1, FRAMES+1):
                frames.append(quantize(frame(overviews[j], levels, slot, k)))
                durations.append(50)
            del levels
        print(f"rendered {hero} to {HEROES[outer]}", flush=True)
    # The endpoint uses the same overview raster as the next leg's first hold.
    for j in range(len(HEROES)):
        frames[(j+1)*(1+FRAMES)-1] = frames[((j+1)%len(HEROES))*(1+FRAMES)].copy()
    sheet.save(root / "five_boards.jpg", quality=94)
    gif = root / "spider_iron_captain_hulk_thor_zoom_out.gif"
    tmp = gif.with_suffix(".tmp.gif")
    frames[0].save(tmp, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   disposal=1, optimize=False)
    with Image.open(tmp) as film:
        first = film.convert("RGB").tobytes()
        elapsed, stops = 0, []
        for k in range(film.n_frames):
            film.seek(k)
            delay = film.info["duration"]
            if delay >= 5000:
                stops.append(film.convert("RGB").tobytes())
            elapsed += delay
        expected = [frames[j*(1+FRAMES)].convert("RGB").tobytes() for j in range(len(HEROES))]
        if (stops != expected or elapsed != 50000 or film.info["loop"] != 0 or
                film.convert("RGB").tobytes() != first):
            raise RuntimeError("Encoded animation failed order, timing or loop validation")
        count = film.n_frames
    tmp.replace(gif)
    write_json(root / "render.json", dict(order=[*HEROES, HEROES[0]], edges=edges, frames=count,
               duration_ms=elapsed, size=SIDE, loop_verified=True, hold_palette_mae_128=palette_errors,
               sha256=digest(gif)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
