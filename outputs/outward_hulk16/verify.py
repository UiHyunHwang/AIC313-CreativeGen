"""Check all five stops, retained artwork, and the greens in the actual output GIF."""
from pathlib import Path
import json
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import CLIPModel,CLIPProcessor
from sdxl.hero_cycle import CLIP_ID,CLIP_REV
from sdxl.hero_cycle_render import digest
from outputs.outward_palette16 import camera
from outputs.outward_thor16.verify import THOR_TEXT,stops
from outputs.outward_hulk16.render_cycle import ROOT,OUT,PRIOR,HEROES,GIF,PREVIOUS_GIF,read
Image.MAX_IMAGE_PIXELS=16384**2
HULK_TEXT=(["Marvel's Incredible Hulk, a gigantic muscular green superhero","Hulk with green skin, black hair and clenched fists"],
           ["an empty industrial background","a small distant speck","just green foliage without a character"])

@torch.inference_mode()
def scores(views,slots):
    model=CLIPModel.from_pretrained(CLIP_ID,revision=CLIP_REV,local_files_only=True).eval()
    proc=CLIPProcessor.from_pretrained(CLIP_ID,revision=CLIP_REV,local_files_only=True,use_fast=False)
    result={}
    for j,(pos,neg) in ((0,HULK_TEXT),(4,THOR_TEXT)):
        texts=F.normalize(model.get_text_features(**proc(text=pos+neg,return_tensors="pt",padding=True)))
        ims=[]
        for i in range(256):
            if i==slots[j]:continue
            x,y=(i%16)*64,(i//16)*64
            ims.append(views[j].crop((x,y,x+64,y+64)).resize((44,44),Image.Resampling.LANCZOS))
        margins=[]
        for n in range(0,len(ims),8):
            f=F.normalize(model.get_image_features(**proc(images=ims[n:n+8],return_tensors="pt")))
            sim=f@texts.T
            margins.extend((sim[:,:len(pos)].amax(1)-sim[:,len(pos):].amax(1)).tolist())
        assert len(margins)==255
        result[HEROES[j-1]]=dict(count=len(margins),positive=int((np.array(margins)>0).sum()),mean=float(np.mean(margins)))
    return result

def main():
    r=read(ROOT/"render.json");gen=read(ROOT/"generation.json")
    assert digest(ROOT/GIF)==r["gif"]["gif_sha256"]
    assert digest(PRIOR/PREVIOUS_GIF)==read(PRIOR/"verified.json")["gif"]["gif_sha256"]
    before=stops(PRIOR/PREVIOUS_GIF);after=stops(ROOT/GIF)
    assert len(before)==4 and len(after)==5 and r["gif"]["duration_ms"]==50020
    slots=[(e["row"]-1)*16+e["column"]-1 for e in r["edges"]]
    diffs={}
    for j in (1,2,3):
        mask=np.ones(after[j].shape[:2],dtype=bool)
        row,col=divmod(slots[j],16);mask[44*row:44*(row+1),44*col:44*(col+1)]=False
        diff=np.abs(after[j]-before[j])
        diffs[HEROES[j]]=dict(mean_abs_8bit=float(diff[mask].mean()),identical_fraction=float(np.all(diff[mask]==0,axis=1).mean()))
    views=[]
    for hero in HEROES:
        with Image.open(ROOT/(hero+"_16x16.jpg")) as im:
            assert im.size==(16384,16384);im.verify()
        with Image.open(ROOT/(hero+"_overview.png")) as im:views.append(im.convert("RGB"))
    true_hulk=camera.project_layers([[v] for v in views],slots,4,(0.,0.,1.))
    with Image.open(PRIOR/PREVIOUS_GIF) as f:
        original=Image.new("P",(1,1));original.putpalette(f.getpalette())
    old_quant=np.array(true_hulk.quantize(palette=original,dither=Image.Dither.FLOYDSTEINBERG).convert("RGB"),dtype=np.int16)
    truth=np.array(true_hulk,dtype=np.int16)
    red,green,blue=np.moveaxis(truth,-1,0);mask=(green>=red-5)&(green-blue>12)
    palette_error=dict(pixels=int(mask.sum()),old=float(np.abs(old_quant-truth)[mask].mean()),new=float(np.abs(after[4]-truth)[mask].mean()))
    assert mask.any() and palette_error["new"]<palette_error["old"]
    result=dict(grid=16,generated_native_leaves=512,candidates=4,hero_order=r["gif"]["order"],preservation=r["preservation"],
                encoded_holds_outside_link=diffs,green_rgb_mae_8bit=palette_error,clip=scores(views,slots),
                clip_model=CLIP_ID,clip_revision=CLIP_REV,gif=r["gif"],peak_generation_gib=gen["peak_reserved_gib"])
    (ROOT/"verified.json").write_text(json.dumps(result,indent=2)+"\n")
    project=OUT.parent;shared=OUT/"outward_palette16/camera.py"
    with ZipFile(PRIOR/"reproduction.zip") as old,ZipFile(ROOT/"reproduction.zip","w",ZIP_DEFLATED,compresslevel=3) as z:
        for name in old.namelist():
            if name!=str(shared.relative_to(project)):z.writestr(name,old.read(name))
        z.write(shared,shared.relative_to(project))
        for p in sorted(ROOT.rglob("*")):
            if p.is_file() and p.suffix in (".py",".md",".json"):z.write(p,p.relative_to(project))
    with ZipFile(ROOT/"reproduction.zip") as z:assert z.testzip() is None
    print(result,flush=True)
if __name__=="__main__":main()
