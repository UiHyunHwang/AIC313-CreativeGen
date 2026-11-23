"""Replay the two non-generative CLIP rankings from the recorded verifier texts."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model_id = "openai/clip-vit-base-patch32"
revision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

def rank_tiles(tiles, positive, negative):
    processor=CLIPProcessor.from_pretrained(model_id,revision=revision,local_files_only=True,use_fast=False)
    model=CLIPModel.from_pretrained(model_id,revision=revision,local_files_only=True).eval()
    tiles=sorted(tiles)
    with torch.inference_mode():
        text=processor(text=positive+negative,return_tensors="pt",padding=True)
        prompts=torch.nn.functional.normalize(model.get_text_features(**text),dim=-1)
        scores=[]
        for start in range(0,len(tiles),8):
            paths=tiles[start:start+8]
            images=[]
            for p in paths:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            values=[]
            for views in [images,[im.resize((32,32),Image.Resampling.BOX) for im in images]]:
                inputs=processor(images=views,return_tensors="pt")
                features=torch.nn.functional.normalize(model.get_image_features(**inputs),dim=-1)
                sims=features@prompts.T
                values.append(sims[:,:len(positive)].mean(dim=1)-sims[:,len(positive):].amax(dim=1))
            scores.extend({"tile":p.stem,"full":float(a),"thumbnail":float(b),"score":float((a+b)/2)}
                          for p,a,b in zip(paths,*values))
    scores.sort(key=lambda item:(-item["score"],item["tile"]))
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["spider", "captain"])
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    recorded = json.loads(Path(__file__).with_name(args.stage+"_selection.json").read_text())
    paths = sorted((args.run/"tiles").glob("*.png"))
    assert len(paths) == 256
    scores = rank_tiles(paths, recorded["positive"], recorded["negative"])
    args.output.write_text(json.dumps(scores, indent=2))
    assert scores[0]["tile"] == recorded["ranked"][0]["tile"], "Zoom coordinate differs from recorded choice"
