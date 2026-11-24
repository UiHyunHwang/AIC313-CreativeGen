"""Score all generated tiles and the five overview images after sampling."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from .hero_cycle import CLIP_ID, CLIP_REV, write_json
from .joint_cycle import GRID, HEROES, TILE
from .joint_zoom import error


@torch.inference_mode()
def verify(root):
    names = ("Spider-Man", "Iron Man", "Captain America", "the Hulk", "Thor")
    labels = [f"a picture of {name}" for name in names]
    opts = dict(revision=CLIP_REV, local_files_only=True)
    model = CLIPModel.from_pretrained(CLIP_ID, **opts).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID, use_fast=False, **opts)
    classes = F.normalize(model.get_text_features(**proc(text=labels, padding=True, return_tensors="pt")))

    def scores(images):
        inputs = proc(images=images, return_tensors="pt")
        return F.normalize(model.get_image_features(**inputs)) @ classes.T

    Image.MAX_IMAGE_PIXELS = (GRID*TILE)**2
    slots = json.loads((root / "generation.json").read_text())["slots"]
    results = {}
    for j, hero in enumerate(HEROES):
        folder = root / hero
        with Image.open(folder / "overview.png") as view, Image.open(folder / "target.png") as target:
            seen = scores([view.resize((32,32), Image.Resampling.BOX), target]).argmax(1).tolist()
            record = dict(overview_label=names[seen[0]], target_label=names[seen[1]],
                          target_mae_16=error(view, target, 16))
        winners = []
        with Image.open(folder / "board.png") as board:
            for row in range(GRID):
                tiles = [board.crop((col*TILE,row*TILE,(col+1)*TILE,(row+1)*TILE))
                         .resize((256,256), Image.Resampling.LANCZOS) for col in range(GRID)]
                winners.extend(scores(tiles).argmax(1).tolist())
        expected = (j-1) % len(names)
        record.update(tile_subject=names[expected], portal_label=names[winners[slots[j]]],
                      correct_tiles=winners.count(expected), total_tiles=len(winners))
        results[hero] = record
        print(hero, record, flush=True)
    write_json(root / "quality.json", dict(clip=CLIP_ID, revision=CLIP_REV, labels=labels, boards=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root)
