"""Evaluate the new ordinary leaves and verify the delivered four-stop film."""
from pathlib import Path
import json
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from transformers import CLIPModel,CLIPProcessor
from sdxl.hero_cycle import CLIP_ID,CLIP_REV
from outputs.outward_palette16.quality import TEXT
from outputs.outward_thor16.render_cycle import ROOT,OUT,PRIOR,HEROES,GIF,read
from sdxl.hero_cycle_render import digest
Image.MAX_IMAGE_PIXELS=16384**2
THOR_TEXT=(["Thor in a red cape and Asgardian armor with long blond hair","Marvel's Thor wielding Mjolnir, bearded and wearing a red cape"],
           ["a hammer alone without a person","an empty landscape","a tiny distant speck"])
def stops(path):
    result=[]
    with Image.open(path) as im:
        for i in range(im.n_frames):
            im.seek(i)
            if im.info["duration"]>=1000:result.append(np.array(im.convert("RGB"),dtype=np.int16))
        assert np.array_equal(result[0],np.asarray(im.convert("RGB")))
    return result
@torch.inference_mode()
def scores(gen,edges):
    model=CLIPModel.from_pretrained(CLIP_ID,revision=CLIP_REV,local_files_only=True).eval()
    proc=CLIPProcessor.from_pretrained(CLIP_ID,revision=CLIP_REV,local_files_only=True,use_fast=False)
    summary={}
    for j,(pos,neg) in ((0,THOR_TEXT),(3,TEXT[0])):
        stage=gen["stages"][j];label=stage["leaf"];edge=edges[j]
        slot=(edge["row"]-1)*16+edge["column"]-1
        paths=[OUT/stage["directory"]/r["file"] for r in stage["tiles"] if r["index"]!=slot]
        assert len(paths)==255
        text=F.normalize(model.get_text_features(**proc(text=pos+neg,return_tensors="pt",padding=True)))
        margins=[]
        for i in range(0,len(paths),8):
            ims=[]
            for p in paths[i:i+8]:
                with Image.open(p) as im:ims.append(im.convert("RGB").reduce(16).resize((44,44),Image.Resampling.LANCZOS))
            feat=F.normalize(model.get_image_features(**proc(images=ims,return_tensors="pt")))
            sims=feat@text.T
            margins.extend((sims[:,:len(pos)].amax(1)-sims[:,len(pos):].amax(1)).tolist())
        summary[label]=dict(count=len(paths),positive=int((np.array(margins)>0).sum()),mean=float(np.mean(margins)))
        print("44-pixel CLIP",label,summary[label],flush=True)
    return summary
def main():
    report=read(ROOT/"render.json");gen=read(ROOT/"generation.json")
    gif=ROOT/GIF
    assert digest(gif)==report["gif"]["gif_sha256"]
    old=stops(PRIOR/"spider_iron_captain_zoom_out.gif");new=stops(gif)
    assert len(new)==4 and len(old)==3
    difference={}
    for j in (1,2):
        edge=report["edges"][j]
        delta=np.abs(new[j]-old[j])
        mask=np.ones(delta.shape[:2],dtype=bool)
        r,c=edge["row"]-1,edge["column"]-1
        mask[44*r:44*(r+1),44*c:44*(c+1)]=False
        difference[HEROES[j]]=dict(mean_abs_8bit=float(delta[mask].mean()),identical_fraction=float(np.all(delta[mask]==0,axis=1).mean()))
    for hero in HEROES:
        with Image.open(ROOT/(hero+"_16x16.jpg")) as im:
            assert im.size==(16384,16384);im.verify()
    stats=scores(gen,report["edges"])
    record=dict(grid=16,new_native_leaves=512,new_target_samples=4,preservation=report["preservation"],
                retained_gif_holds_outside_link=difference,thumbnail_scores=stats,
                clip_model=CLIP_ID,clip_revision=CLIP_REV,gif=report["gif"],peak_reserved_gib=gen["peak_reserved_gib"])
    (ROOT/"verified.json").write_text(json.dumps(record,indent=2)+"\n")
    project=OUT.parent;camera=OUT/"outward_palette16/camera.py"
    with ZipFile(PRIOR/"reproduction.zip") as previous, ZipFile(ROOT/"reproduction.zip","w",ZIP_DEFLATED,compresslevel=3) as bundle:
        for name in previous.namelist():
            if name!=str(camera.relative_to(project)):bundle.writestr(name,previous.read(name))
        bundle.write(camera,camera.relative_to(project))
        for p in sorted(ROOT.rglob("*")):
            if p.is_file() and p.suffix in (".py",".md",".json"):
                bundle.write(p,p.relative_to(project))
    with ZipFile(ROOT/"reproduction.zip") as bundle:assert bundle.testzip() is None
    print("Verified",record,flush=True)
if __name__=="__main__":main()
