"""Select a zoom coupling profile on the matched Captain→Hulk pilot."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from .evaluate_joint_zoom import CLIP_ID, CLIP_REV, LABELS
from .hero_cycle import write_json
from .joint_zoom import error


@torch.inference_mode()
def select(folder):
    report = json.loads((folder / "run.json").read_text())
    opts = dict(revision=CLIP_REV, local_files_only=True)
    model = CLIPModel.from_pretrained(CLIP_ID, **opts).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID, use_fast=False, **opts)
    labels = F.normalize(model.get_text_features(**proc(text=LABELS, padding=True, return_tensors="pt")))
    scores = {}
    for arm, metrics in report["arms"].items():
        with Image.open(folder / arm / "portal.png") as portal, Image.open(folder / arm / "child.png") as child:
            features = F.normalize(model.get_image_features(**proc(images=portal, return_tensors="pt")))
            sims = (features @ labels.T)[0]
            margin = float(sims[1] - sims[[0, 2]].max())
            loss = error(portal, child, 128)
            scores[arm] = dict(captain_margin=margin, cut_mae_128=loss, cut_mae_32=metrics["portal_child_mae_32"],
                               target_mae=max(metrics["parent_hulk_mae_16"], metrics["child_captain_mae_16"]))
    eligible = [a for a, s in scores.items() if s["captain_margin"] > 0 and s["target_mae"] <= scores["control"]["target_mae"] + .015]
    selected = min(eligible, key=lambda a: scores[a]["cut_mae_32"] + scores[a]["cut_mae_128"])
    write_json(folder / "cycle_selection.json", dict(seed=report["seed"], scores=scores, selected=selected,
               rule="Positive Captain CLIP margin and max target MAE within 0.015 of control; minimize 32+128 cut MAE."))
    print(selected, scores[selected])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    select(parser.parse_args().folder)
