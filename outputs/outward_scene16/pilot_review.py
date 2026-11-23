"""Evaluate 16 paired generative trajectories in stratified parent patches."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from transformers import CLIPModel,CLIPProcessor
from outputs.outward_palette16.quality import TEXT,HEROES
from sdxl.hero_cycle import CLIP_ID,CLIP_REV
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
def read(p):return json.loads(p.read_text())
def rgb(im,side):return np.asarray(im.resize((side,side),Image.Resampling.BOX),dtype=float)/255
def put(path,value):path.write_text(json.dumps(value,indent=2)+"\n")
@torch.inference_mode()
def main():
    rows=read(ROOT/"pilot.json")
    proc=CLIPProcessor.from_pretrained(CLIP_ID,revision=CLIP_REV,use_fast=False,local_files_only=True)
    model=CLIPModel.from_pretrained(CLIP_ID,revision=CLIP_REV,local_files_only=True).eval()
    def text(labels):return F.normalize(model.get_text_features(**proc(text=labels,padding=True,return_tensors="pt")))
    set_text=text(["a film still of a character inside a detailed physical setting","a character standing in a scene with foreground and background depth",
                   "a faint isolated character on a plain studio backdrop","a small cutout of a person on an empty background"])
    hero_text=[text(p+n) for p,n in TEXT]
    tar=[Image.open(OUT/p).convert("RGB") for p in read(ROOT/"targets.json")["selected"]]
    sheet=Image.new("RGB",(4*280,4*164),"#faf9f5");draw=ImageDraw.Draw(sheet)
    for k,row in enumerate(rows):
        j=HEROES.index(row["hero"]);r,c=divmod(row["index"],16);x,y=k%4*280,k//4*164
        draw.text((x+2,y+3),f"{HEROES[j]}  R{r+1} C{c+1}  previous / scene",fill="#252525")
        ref=tar[j].crop((64*c,64*r,64*(c+1),64*(r+1)))
        for col,mode in enumerate(("control","scene")):
            with Image.open(OUT/row[mode]) as f:im=f.convert("RGB")
            sheet.paste(im.resize((136,136),Image.Resampling.LANCZOS),(x+138*col,y+22))
            thumb=im.resize((64,64),Image.Resampling.BOX).resize((44,44),Image.Resampling.LANCZOS)
            feats=F.normalize(model.get_image_features(**proc(images=[im,thumb],return_tensors="pt")))
            s=feats[0]@set_text.T;h=feats[1]@hero_text[j].T
            row[mode+"_metrics"]=dict(scene=float(s[:2].amax()-s[2:].amax()),identity44=float(h[:2].amax()-h[2:].amax()),
                                       color=float(np.abs(rgb(ref,4)-rgb(im,4)).mean()))
    sheet.save(ROOT/"pilot_comparison.jpg",quality=95)
    stats={}
    for key in ("scene","identity44","color"):
        prev=np.array([r["control_metrics"][key] for r in rows]);new=np.array([r["scene_metrics"][key] for r in rows])
        stats[key]=dict(control=float(prev.mean()),scene=float(new.mean()),delta=float((new-prev).mean()),increased=int((new>prev).sum()))
    adopt=(stats["scene"]["delta"]>.005 and stats["scene"]["increased"]>len(rows)//2 and
           stats["identity44"]["delta"]>-.018 and stats["color"]["delta"]<.04)
    put(ROOT/"pilot_review.json",dict(adopt=adopt,metrics=stats,rows=rows))
    print(adopt,stats,flush=True)
if __name__=="__main__":main()
