"""Resolve four recursive grids and render their four forward zooms."""
import argparse
import hashlib
import json
from pathlib import Path
import math
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageChops

from sdxl.hero_cycle_render import read,digest,cell,xy,BOX,LANCZOS,SIDE,project

def pixel_hash(im):
    return hashlib.sha256(im.tobytes()).hexdigest()

def rgb(im, size):
    return np.asarray(im.resize((size,size),BOX),dtype=float)/255

def thumbnail(run, replacements):
    with Image.open(run/"mosaic.png") as im:
        assert im.size == (16384,16384)
        result = im.resize((1024,1024),BOX)
    for i, path in replacements.items():
        with Image.open(path) as im:
            result.paste(im.resize((64,64),BOX),xy(i,64))
    return result

def fit_return(opening, fourth, target):
    incoming = rgb(opening,64)
    signature = rgb(opening,4)
    board = rgb(fourth,1024)
    costs = []
    for i in range(256):
        r,c = divmod(i,16)
        x,y = xy(i,64)
        boundaries = []
        if r: boundaries.append(np.abs(incoming[0]-board[y-1,x:x+64]).mean())
        if r<15: boundaries.append(np.abs(incoming[-1]-board[y+64,x:x+64]).mean())
        if c: boundaries.append(np.abs(incoming[:,0]-board[y:y+64,x-1]).mean())
        if c<15: boundaries.append(np.abs(incoming[:,-1]-board[y:y+64,x+64]).mean())
        seam = float(np.mean(boundaries))
        color = float(np.abs(signature-rgb(cell(target,i),4)).mean())
        costs.append(dict(index=i,seam=seam,color=color,cost=.65*seam+.35*color))
    return min(costs,key=lambda v:(v["cost"],v["index"]))["index"], costs

def draw(levels, slots, node, progress):
    if progress==1:
        node=(node+1)%4
        progress=0.
    slot = slots[node]
    ease = progress**3*(10-15*progress+6*progress**2)
    width=16**(-ease)
    scale=(1-width)/(1-1/16)
    x,y=xy(slot,1/16)
    bounds=x*scale,y*scale,width
    frame=Image.new("RGB",(SIDE,SIDE))
    project(frame,levels[node],bounds,0,0,1)
    project(frame,levels[(node+1)%4],bounds,x,y,1/16)
    return frame

def animate(root, levels, slots):
    atlas=Image.new("RGB",(800,1100))
    for node in range(4):
        for k in range(19):
            idx=node*19+k
            im=draw(levels,slots,node,k/18)
            atlas.paste(im.resize((100,100),LANCZOS),xy_atlas(idx))
    palette=atlas.crop((0,0,800,1000)).quantize(256,method=Image.Quantize.MEDIANCUT)
    def frame(node,progress):
        return draw(levels,slots,node,progress).quantize(palette=palette,dither=Image.Dither.FLOYDSTEINBERG)
    def same(a,b):
        return ImageChops.difference(a,b).getbbox() is None
    stops=[frame(i,0) for i in range(4)]
    frames,durations=[],[]
    for node in range(4):
        frames.append(stops[node]);durations.append(5500)
        movement=[frame(node,k/91) for k in range(1,91)]
        while movement and same(stops[node],movement[0]): movement.pop(0)
        while movement and same(stops[(node+1)%4],movement[-1]): movement.pop()
        assert movement
        frames.extend(movement)
        ticks=np.rint(np.linspace(0,450,len(movement)+1)).astype(int)
        durations.extend((10*np.diff(ticks)).tolist())
    frames.append(frame(3,1));durations.append(20)
    assert same(frames[0],frames[-1])
    gif=root/"inward_cycle.gif"
    frames[0].save(gif,save_all=True,append_images=frames[1:],optimize=False,loop=0,disposal=1,duration=durations)
    ms=0;pauses=[];actual=[]
    with Image.open(gif) as im:
        assert im.info["loop"]==0
        for i in range(im.n_frames):
            im.seek(i);duration=im.info["duration"]
            if duration>=1000:
                pauses.append(duration)
                actual.append(im.convert("RGB"))
            ms+=duration
        wrap=im.convert("RGB");n=im.n_frames
    assert ms==40020 and pauses==[5500]*4 and same(actual[0],wrap)
    assert all(same(got,want.convert("RGB")) for got,want in zip(actual,stops))
    return dict(duration_ms=ms,frames=n,holds_ms=pauses,loop_forever=True,first_last_identical=True,sha256=digest(gif))

def xy_atlas(index):
    return index%8*100,index//8*100

def main(root,prior):
    base=root.parent/"hero_cycle_varied"
    cap=root.parent/"captain_refined"
    archived=root.parent/"hero_cycle"
    raw=[base/p for p in read(base/"report.json")["raw_runs"]]
    extra=[p.parent for p in (prior/"new_board").glob("*/manifest.json") if read(p)["status"]=="complete"]
    assert len(extra)==1
    raw+=extra
    old=read(prior/"quality.json")
    new=read(root/"spider.json")
    overrides=[{v["index"]:(archived/v["file"]).with_suffix(".png") for v in old["selected"]},
        {v["index"]:(cap/v["file"]).with_suffix(".png") for v in read(cap/"refinement.json")["selected"] if v["seed"]!=43},{},{}]
    overrides[0].update({v["index"]:(root/v["file"]).with_suffix(".png") for v in new["selected"]})
    backup=read(root/"final_selection.json")["selected"]
    overrides[0].update({v["index"]:(archived/v["file"]).with_suffix(".png") for v in backup})
    prev=read(prior/"report.json")["edges"]
    assert [(p["parent"],p["child"]) for p in prev]==[(0,1),(1,2),(2,3),(2,0)]
    indices=[(p["row"]-1)*16+p["column"]-1 for p in prev]
    slots,old_return=indices[:3],indices[3]
    assert all(idx not in table for idx,table in zip(slots,overrides))
    templates=[thumbnail(run,repl) for run,repl in zip(raw,overrides)]
    with Image.open(raw[3]/"target.png") as target:
        opening=templates[0].copy()
        opening.paste(templates[1].resize((64,64),BOX),xy(slots[0],64))
        back,costs=fit_return(opening,templates[3],target)
    slots.append(back)
    thumbs=templates
    for iteration in range(1,17):
        updated=[im.copy() for im in templates]
        for j,i in enumerate(slots):
            updated[j].paste(thumbs[(j+1)%4].resize((64,64),BOX),xy(i,64))
        stable=all(a.tobytes()==b.tobytes() for a,b in zip(thumbs,updated))
        thumbs=updated
        if stable:break
    assert stable
    levels,edges=[],[]
    restored=None
    for j,(slot,run) in enumerate(zip(slots,raw)):
        with Image.open(run/"mosaic.png") as im: board=im.convert("RGB")
        for i,path in overrides[j].items():
            with Image.open(path) as tile:board.paste(tile,xy(i,1024))
        child=(j+1)%4
        board.paste(thumbs[child],xy(slot,1024))
        assert board.resize((1024,1024),BOX).tobytes()==thumbs[j].tobytes()
        assert cell(board,slot).tobytes()==thumbs[child].tobytes()
        edges.append(dict(parent=j,child=child,row=slot//16+1,column=slot%16+1,child_pixel_sha256=pixel_hash(thumbs[child])))
        if j==2:
            name=f"r{old_return//16:03}_c{old_return%16:03}.png"
            with Image.open(run/"tiles"/name) as im:
                assert cell(board,old_return).tobytes()==im.tobytes()
            restored=dict(row=old_return//16+1,column=old_return%16+1,tile=name,png_sha256=digest(run/"tiles"/name))
        thumbs[j].save(root/f"board_{j}_overview.png")
        board.save(root/f"board_{j}.png",compress_level=2)
        levels.append([board.reduce(k) for k in (16,8,4,2)])
        del board
        print("Verified board",j,edges[-1],flush=True)
    video=animate(root,levels,slots)
    report=dict(edges=edges,restored_captain_tile=restored,gif=video,return_costs=costs,iterations=iteration)
    (root/"report.json").write_text(json.dumps(report,indent=2))
    with ZipFile(root/"reproduction.zip","w",compression=ZIP_DEFLATED) as out:
        out.write(prior/"reproduction.zip","prior.zip")
        for name in ("refine_spider.py","render_cycle.py","check_archived.py","README.md","spider.json","archived_scores.json","final_selection.json","report.json"):
            out.write(root/name,"inward/"+name)
    with ZipFile(root/"reproduction.zip") as z:assert z.testzip() is None
    print(video,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("outputs/inward_cycle"))
    p.add_argument("--prior",type=Path,default=Path("outputs/varied_detour_quality"))
    args=p.parse_args();main(args.output,args.prior)
