"""Run from PhotoQuilt with the creativegen Python to select the parent tile."""
from pathlib import Path
import json

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

out=Path(__file__).resolve().parent
base=out.parent/"ironman_spider_16x16_20261119_064518/sdxl_mosaic_20261119_064525_817477_seed42"
model_id="openai/clip-vit-base-patch32"
revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
positive=["a close-up portrait of Spider-Man's red webbed mask and large white eyes",
          "a large, clearly visible Spider-Man facing the viewer, his head and shoulders filling the picture"]
negative=["a distant tiny superhero against a wide background", "a spider insect or spiderweb without a person",
          "an abstract blurry patch with no recognizable subject","a small full-body figure doing a stunt in a landscape"]
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
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, default=base, help="Completed outer 16x16 run")
    base = parser.parse_args().parent_run.resolve()
    tiles=sorted((base/"tiles").glob("*.png"))
    assert len(tiles)==256
    scores=rank_tiles(tiles,positive,negative)
    meta={"parent":str(base.relative_to(out.parents[1])),"method":"mean positive cosine minus maximum negative cosine, averaged over full tile and 32px thumbnail","model_id":model_id,"revision":revision,"positive":positive,"negative":negative,"ranked":scores}
    (out/"selection.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("Selected",scores[0],flush=True)
