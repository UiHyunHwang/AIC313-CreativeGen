"""Offline evaluation and raw zoom preview for a completed joint_zoom run."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from .hero_cycle import CLIP_ID, CLIP_REV

LABELS = ("a picture of the Hulk", "a picture of Captain America", "a picture of Iron Man")


def tile_box(row, col, side):
    left, top = col * side, row * side
    return left, top, left + side, top + side


def zoom(parent, child, grid, row, col, path):
    """Cut directly from the child overview to the true parent crop, then pull back."""
    side = parent.width
    frames = [child.resize((512, 512), Image.Resampling.LANCZOS)]
    for step in range(37):
        fraction = step / 36
        width = side * grid ** (fraction - 1)
        # Interpolate the crop from the tile to the whole board at constant log scale.
        x = col / (grid - 1) * (side - width)
        y = row / (grid - 1) * (side - width)
        frames.append(parent.resize((512, 512), Image.Resampling.LANCZOS,
                                    box=(x, y, min(side, x + width), min(side, y + width))))
    frames[0].save(path, save_all=True, append_images=frames[1:], lossless=True,
                   duration=[1400, 700] + [80] * 35 + [1400], loop=0)


@torch.inference_mode()
def evaluate(folder):
    report = json.loads((folder / "run.json").read_text())
    grid = report["grid"]
    row, col = report["slot_zero_based"]
    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS, (grid * 1024) ** 2)
    options = dict(revision=CLIP_REV, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(CLIP_ID, use_fast=False, **options)
    model = CLIPModel.from_pretrained(CLIP_ID, **options).eval()
    text = processor(text=LABELS, return_tensors="pt", padding=True)
    labels = F.normalize(model.get_text_features(**text))

    def scores(images):
        data = processor(images=images, return_tensors="pt")
        return F.normalize(model.get_image_features(**data)) @ labels.T

    results = dict(model=CLIP_ID, revision=CLIP_REV, labels=LABELS, arms={})
    side = 256
    strip_height = side + 20
    contact = Image.new("RGB", (3 * side, len(report["arms"]) * strip_height))
    captions = ImageDraw.Draw(contact)
    for index, arm in enumerate(report["arms"]):
        root = folder / arm
        with Image.open(root / "parent.png") as parent, Image.open(root / "child.png") as child:
            values = {}
            for level, (name, image) in enumerate((("parent", parent), ("child", child))):
                tile_side = image.width // grid
                boxes = (tile_box(r, c, tile_side) for r in range(grid) for c in range(grid))
                crops = [image.crop(box).resize((side, side), Image.Resampling.LANCZOS) for box in boxes]
                features = torch.cat([scores(crops[i:i + 8]) for i in range(0, len(crops), 8)])
                views = [image.resize((n, n), Image.Resampling.BOX) for n in (512, 32)]
                overview, thumbnail = scores(views).tolist()
                values[name] = dict(overview_similarities=overview, thumbnail_similarities=thumbnail,
                                    correct_tiles=int((features.argmax(1) == level + 1).sum()),
                                    total_tiles=len(crops))
            box = tile_box(row, col, parent.width // grid)
            with Image.open(root / "portal.png") as portal:
                if parent.crop(box).tobytes() != portal.tobytes():
                    raise ValueError("Exported parent crop differs from its sampled tile")
                if child.resize(portal.size, Image.Resampling.BOX).tobytes() == portal.tobytes():
                    raise ValueError("Portal unexpectedly equals the reduced child board")
                values["portal_similarities"] = scores([portal])[0].tolist()
                panels = (("child", child), ("parent tile", portal), ("parent", parent))
                for col_index, (label, view) in enumerate(panels):
                    x, y = col_index * side, index * strip_height
                    captions.text((x + 3, y + 2), f"{arm} / {label}", fill="white")
                    contact.paste(view.resize((side, side), Image.Resampling.BOX), (x, y + 20))
            zoom(parent, child, grid, row, col, root / "zoom.webp")
        results["arms"][arm] = values
    contact.save(folder / "comparison.jpg", quality=94)
    (folder / "evaluation.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    evaluate(parser.parse_args().folder)
