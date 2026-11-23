"""Audit the edited board and report Hulk color compatibility without making a Hulk film."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.nn import functional as F
from transformers import CLIPModel,CLIPProcessor
from sdxl.hero_cycle import CLIP_ID,CLIP_REV
from outputs.outward_palette16.quality import TEXT
from outputs.outward_thor16.verify import stops
from outputs.outward_thor_iron16.targets import ROOT,OUT,PRIOR,HEROES,open_rgb,palette,balanced_match,signature
from outputs.outward_thor_iron16.render import GIF,core
from outputs.outward_thor_iron16 import hulk_placement
Image.MAX_IMAGE_PIXELS=16384**2

@torch.inference_mode()
def quality():
    model=CLIPModel.from_pretrained(CLIP_ID,revision=CLIP_REV,local_files_only=True).eval()
    proc=CLIPProcessor.from_pretrained(CLIP_ID,revision=CLIP_REV,use_fast=False,local_files_only=True)
    pos,neg=TEXT[1]
    text=F.normalize(model.get_text_features(**proc(text=pos+neg,padding=True,return_tensors="pt")))
    summaries={}
    for key,root in (("previous",PRIOR),("edited",ROOT)):
        sel=core.read_json(root/"targets.json")
        own=open_rgb(OUT/sel["selected"][1]);spider=open_rgb(OUT/sel["selected"][0])
        image=open_rgb(root/"iron_man_overview.png");edge=core.read_json(root/"render.json")["edges"][1]
        slot=(edge["row"]-1)*16+edge["column"]-1
        margins=[];ims=[]
        for i in range(256):
            if i==slot:continue
            r,c=divmod(i,16)
            ims.append(image.crop((64*c,64*r,64*c+64,64*r+64)).resize((44,44),Image.Resampling.LANCZOS))
        for i in range(0,len(ims),8):
            feat=F.normalize(model.get_image_features(**proc(images=ims[i:i+8],return_tensors="pt")))
            sims=feat@text.T
            margins.extend((sims[:,:len(pos)].amax(1)-sims[:,len(pos):].amax(1)).tolist())
        summaries[key]=dict(mosaic_spider_palette=balanced_match(palette(image),palette(spider)),
             mosaic_rgb_mae64=float(np.abs(signature(image,64)-signature(own,64)).mean()),
             leaf_identity_mean=float(np.mean(margins)),leaf_identity_positive=int((np.array(margins)>0).sum()),n=len(margins))
    return summaries

def main():
    hulk_placement.main()
    report=core.read_json(ROOT/"render.json");gen=core.read_json(ROOT/"generation.json")
    assert core.digest(ROOT/GIF)==report["gif"]["gif_sha256"] and report["gif"]["order"]==[*HEROES,HEROES[0]]
    old,edited=stops(PRIOR/GIF),stops(ROOT/GIF)
    assert len(old)==len(edited)==4
    difference={}
    for j in (0,2,3):
        row=report["edges"][j];r,c=row["row"]-1,row["column"]-1
        mask=np.ones(old[j].shape[:2],dtype=bool);mask[44*r:44*(r+1),44*c:44*(c+1)]=False
        difference[HEROES[j]]=float(np.abs(old[j]-edited[j])[mask].mean())
    with Image.open(ROOT/"iron_man_16x16.jpg") as im:
        assert im.size==(16384,16384);im.verify()
    stats=quality()
    verified=dict(native_side=16384,generated_tiles=256,sampled_targets=6,quality=stats,clip=[CLIP_ID,CLIP_REV],
        preservation=report["preserved"],encoded_rgb_mae_outside_retained_links=difference,gif=report["gif"],peak_gpu_gib=gen["peak_reserved_gib"])
    core.write_json(ROOT/"verified.json",verified)
    project=OUT.parent
    # Include the two specific team-generated reference images used only for the numerical Hulk analysis.
    hulkroot=OUT/"outward_hulk16";src=core.read_json(hulkroot/"targets.json")["selected"][4]
    extra=[hulkroot/"targets.json",OUT/src,hulkroot/"hulk_overview.png"]
    with ZipFile(PRIOR/"reproduction.zip") as previous,ZipFile(ROOT/"reproduction.zip","w",ZIP_DEFLATED,compresslevel=3) as archive:
        for name in previous.namelist():archive.writestr(name,previous.read(name))
        for p in extra:archive.write(p,p.relative_to(project))
        for p in sorted(ROOT.rglob("*")):
            if p.is_file() and p.suffix in (".py",".md",".json"):archive.write(p,p.relative_to(project))
    with ZipFile(ROOT/"reproduction.zip") as z:assert z.testzip() is None
    print(verified,flush=True)
if __name__=="__main__":main()
