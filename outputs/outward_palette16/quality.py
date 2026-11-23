"""Frozen CLIP comparison of old and new SDXL leaves at the 16x16 GIF display size."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import CLIPModel,CLIPProcessor
from sdxl.hero_cycle import CLIP_ID,CLIP_REV

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
HEROES=("spider_man","iron_man","captain_america")
PRIOR=(OUT/"captain_highlights/spider_man",OUT/"dualscale_reveal/tiles",OUT/"outward_hero_cycle/captain_america")
TEXT=[
    (["Captain America in blue uniform with a star on his chest","Captain America in his blue helmet holding his shield"],["a shield or flag alone without a person","an empty background","a tiny distant speck"]),
    (["Spider-Man in a red and blue suit","Spider-Man wearing a red webbed mask with white eyes"],["an insect or spiderweb without a superhero","a tiny distant speck","an empty background"]),
    (["Iron Man in red and gold armor","Iron Man with a gold faceplate and white glowing eyes"],["an empty background","a tiny distant speck","a red cloth mask with spiderweb markings"])]
@torch.inference_mode()
def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline-only",action="store_true")
    baseline_only=parser.parse_args().baseline_only
    cache=ROOT/"prior_quality.json"
    prior_stats=json.loads(cache.read_text()) if cache.exists() else {}
    model=CLIPModel.from_pretrained(CLIP_ID,revision=CLIP_REV,local_files_only=True).eval()
    proc=CLIPProcessor.from_pretrained(CLIP_ID,revision=CLIP_REV,local_files_only=True,use_fast=False)
    stats={}
    for j,hero in enumerate(HEROES):
        pos,neg=TEXT[j]
        texts=F.normalize(model.get_text_features(**proc(text=pos+neg,return_tensors="pt",padding=True)))
        stats[hero]={}
        for label,folder,n in [("prior",PRIOR[j],8),("current",ROOT/hero,16)]:
            if label=="current" and baseline_only:continue
            if label=="prior" and hero in prior_stats:
                stats[hero][label]=prior_stats[hero];continue
            edgefile=OUT/"captain_highlights/render.json" if n==8 else ROOT/"render.json"
            edge=json.loads(edgefile.read_text())["edges"][j]
            slot=(edge["row"]-1)*n+edge["column"]-1
            paths=[p for i,p in enumerate(sorted(folder.glob("r*_c*.png"))) if i!=slot]
            assert len(paths)==n*n-1
            scores=[]
            for off in range(0,len(paths),8):
                imgs=[]
                for path in paths[off:off+8]:
                    with Image.open(path) as im:
                        imgs.append(im.convert("RGB").resize((64,64),Image.Resampling.BOX).resize((44,44),Image.Resampling.LANCZOS))
                embeds=F.normalize(model.get_image_features(**proc(images=imgs,return_tensors="pt")))
                sims=embeds@texts.T
                scores.extend((sims[:,:len(pos)].amax(1)-sims[:,len(pos):].amax(1)).tolist())
            stats[hero][label]=dict(n=len(scores),positive=int(np.sum(np.array(scores)>0)),mean=float(np.mean(scores)))
            print(hero,label,stats[hero][label],flush=True)
    cache.write_text(json.dumps({h:stats[h]["prior"] for h in HEROES},indent=2)+"\n")
    if baseline_only:return
    (ROOT/"quality.json").write_text(json.dumps(dict(model=CLIP_ID,revision=CLIP_REV,display=44,stats=stats),indent=2)+"\n")
if __name__=="__main__":main()
