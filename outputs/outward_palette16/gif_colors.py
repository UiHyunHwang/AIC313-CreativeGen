"""Measure the color difference at the actual encoded GIF's recursive boundaries."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from outputs.outward_palette16.render16 import ROOT,OUT,BOX,HEROES,read,rgb,features,old_results

def stops(path):
    frames=[]
    with Image.open(path) as gif:
        for i in range(gif.n_frames):
            gif.seek(i)
            if gif.info["duration"]>=1000:frames.append(gif.convert("RGB"))
    assert len(frames)==3
    return frames
def distances(a,b):
    delta=features(a)-features(b)
    return np.array([np.sqrt((delta**2).mean(axis=-1)).mean(),
                     np.sqrt((delta[...,1:]**2).mean(axis=-1)).mean(),
                     np.abs(delta[...,0]).mean()])
def measurements(frames,targets,slots,n):
    result={}
    for j,frame in enumerate(frames):
        side=frame.width//n;t=max(1,side//16);i=slots[j]
        x=(i%n)*side;y=(i//n)*side
        def strip(bounds,s):return rgb(frame.crop(bounds).resize(s,BOX))
        errors=[]
        for ok,inside,outside,shape in (
            (x>0,(x,y,x+t,y+side),(x-t,y,x,y+side),(1,8)),
            (x+side<frame.width,(x+side-t,y,x+side,y+side),(x+side,y,x+side+t,y+side),(1,8)),
            (y>0,(x,y,x+side,y+t),(x,y-t,x+side,y),(8,1)),
            (y+side<frame.height,(x,y+side-t,x+side,y+side),(x,y+side,x+side,y+side+t),(8,1))):
            if ok:errors.append(distances(strip(inside,shape),strip(outside,shape)))
        seam=np.mean(errors,axis=0)
        region=targets[j].resize(frame.size,Image.Resampling.LANCZOS).crop((x,y,x+side,y+side))
        pattern=distances(strip((x,y,x+side,y+side),(4,4)),rgb(region.resize((4,4),BOX)))
        result[HEROES[j]]=dict(border=float(seam[0]),border_chroma=float(seam[1]),border_luma=float(seam[2]),
                               pattern=float(pattern[0]),pattern_chroma=float(pattern[1]))
    return result
def baseline():
    paths=[ROOT/"baseline"/("gif_"+h+".png") for h in HEROES]
    if not all(p.exists() for p in paths):
        for im,path in zip(stops(OUT/"captain_highlights/spider_iron_captain_zoom_out.gif"),paths):im.save(path)
    return [Image.open(p).convert("RGB") for p in paths]
def main():
    oldframes=baseline()
    _,oldt,olds=old_results()
    current=stops(ROOT/"spider_iron_captain_zoom_out.gif")
    report=read(ROOT/"render.json")
    targets=[Image.open(OUT/p).convert("RGB") for p in read(ROOT/"targets.json")["selected"]]
    slots=[(e["row"]-1)*16+e["column"]-1 for e in report["edges"]]
    data=dict(previous=measurements(oldframes,oldt,olds,8),current=measurements(current,targets,slots,16))
    (ROOT/"gif_colors.json").write_text(json.dumps(data,indent=2)+"\n")
    print(data,flush=True)
if __name__=="__main__":main()
