"""Resolve an actual closed three-board mosaic, then film three inward zooms."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image

from sdxl.hero_cycle_render import BOX, LANCZOS, SIDE, digest, project, read

ROOT=Path(__file__).resolve().parent
N=8
HOLD=5500

def xy(i,side):return (i%N*side,i//N*side)

def read_boards(stages):
    boards=[]
    for stage in stages:
        board=Image.new("RGB",(N*1024,N*1024))
        assert [r["index"] for r in stage["tiles"]]==list(range(N*N))
        for r in stage["tiles"]:
            path=(ROOT/r["prefix"]).with_suffix(".png");assert digest(path)==r["png_sha256"]
            with Image.open(path) as im:
                assert im.size==(1024,1024)
                board.paste(im,xy(r["index"],1024))
        boards.append(board)
    return boards

def return_cell(raw,opening,parent):
    side=1024//N
    def rgb(im):return np.asarray(im,dtype=float)/255
    tile=rgb(opening.resize((side,side),BOX))
    signature=rgb(opening.resize((4,4),BOX))
    board=rgb(raw)
    costs=[]
    for i in range(N*N):
        x,y=xy(i,side);bounds=(x,y,x+side,y+side)
        edges=[]
        if x>0:edges.append(np.abs(tile[:,0]-board[y:y+side,x-1]).mean())
        if x+side<1024:edges.append(np.abs(tile[:,-1]-board[y:y+side,x+side]).mean())
        if y>0:edges.append(np.abs(tile[0]-board[y-1,x:x+side]).mean())
        if y+side<1024:edges.append(np.abs(tile[-1]-board[y+side,x:x+side]).mean())
        cost=.55*np.mean(edges)+.45*np.abs(signature-rgb(parent.crop(bounds).resize((4,4),BOX))).mean()
        costs.append(float(cost))
    return int(np.argmin(costs)),costs

def animate(levels,slots):
    def frame(j,f):
        if f==1:
            j=(j+1)%3;f=0.
        ease=f**3*(10-15*f+6*f*f)
        width=N**(-ease);a=(1-width)/(1-1/N)
        x,y=xy(slots[j],1/N)
        bounds=(x*a,y*a,width)
        out=Image.new("RGB",(SIDE,SIDE))
        project(out,levels[j],bounds,0,0,1)
        project(out,levels[(j+1)%3],bounds,x,y,1/N)
        return out
    atlas=Image.new("RGB",(8*100,9*100))
    for j in range(3):
        for k in range(24):
            idx=j*24+k
            atlas.paste(frame(j,k/23).resize((100,100),LANCZOS),(idx%8*100,idx//8*100))
    palette=atlas.quantize(256,method=Image.Quantize.MEDIANCUT)
    def quant(j,f):return frame(j,f).quantize(palette=palette,dither=Image.Dither.FLOYDSTEINBERG)
    stops=[quant(j,0) for j in range(3)]
    frames=[];dur=[]
    for j in range(3):
        frames.append(stops[j]);dur.append(HOLD)
        inner=[quant(j,k/91) for k in range(1,91)]
        while inner and inner[0].tobytes()==stops[j].tobytes():inner.pop(0)
        while inner and inner[-1].tobytes()==stops[(j+1)%3].tobytes():inner.pop()
        assert inner
        frames.extend(inner)
        dur.extend((10*np.diff(np.rint(np.linspace(0,450,len(inner)+1)).astype(int))).tolist())
    frames.append(quant(2,1));dur.append(20)
    gif=ROOT/"spider_iron_captain.gif"
    frames[0].save(gif,save_all=True,append_images=frames[1:],duration=dur,loop=0,optimize=False,disposal=1)
    ms=0;times=[];observed=[]
    with Image.open(gif) as im:
        assert im.info["loop"]==0
        for k in range(im.n_frames):
            im.seek(k);dt=im.info["duration"]
            if dt>=1000:
                times.append((ms,dt));observed.append(im.convert("RGB").tobytes())
            ms+=dt
        wrap=im.convert("RGB").tobytes();count=im.n_frames
    assert times==[(0,HOLD),(10000,HOLD),(20000,HOLD)]
    assert observed==[s.convert("RGB").tobytes() for s in stops]
    assert wrap==observed[0] and ms==30020
    return dict(frames=count,duration_ms=ms,loop_forever=True,first_last_identical=True,stops_ms=times,sha256=digest(gif))

def main():
    stages=read(ROOT/"generation.json")["stages"]
    order=["spider_man","iron_man","captain_america"]
    assert [v["subject"] for v in stages]==order and [v["leaves"] for v in stages]==order[1:]+order[:1]
    boards=read_boards(stages)
    slots=[stage["outgoing"] for stage in stages[:2]]
    for j,i in enumerate(slots):
        assert stages[j]["tiles"][i]["latent_sha256"]==stages[j+1]["source_sha256"]
        assert stages[j]["tiles"][i]["png_sha256"]==digest(ROOT/stages[j+1]["source"])
    templates=[b.resize((1024,1024),BOX) for b in boards]
    opening=templates[0].copy()
    opening.paste(templates[1].resize((128,128),BOX),xy(slots[0],128))
    with Image.open(ROOT/stages[2]["source"]) as parent:
        back,costs=return_cell(templates[2],opening,parent)
    slots.append(back)
    previews=templates
    for iteration in range(1,17):
        update=[im.copy() for im in templates]
        for j,i in enumerate(slots):update[j].paste(previews[(j+1)%3].resize((128,128),BOX),xy(i,128))
        stable=all(a.tobytes()==b.tobytes() for a,b in zip(previews,update))
        previews=update
        if stable:break
    assert stable
    levels=[];edges=[]
    for j,(board,i) in enumerate(zip(boards,slots)):
        child=previews[(j+1)%3];x,y=xy(i,1024)
        board.paste(child,(x,y))
        assert board.crop((x,y,x+1024,y+1024)).tobytes()==child.tobytes()
        assert board.resize((1024,1024),BOX).tobytes()==previews[j].tobytes()
        previews[j].save(ROOT/(order[j]+"_overview.png"))
        board.save(ROOT/(order[j]+"_mosaic.png"),compress_level=2)
        levels.append([board.reduce(k) for k in (8,4,2,1)])
        edges.append(dict(parent=order[j],child=order[(j+1)%3],row=i//N+1,column=i%N+1,
                          child_pixels_sha256=hashlib.sha256(child.tobytes()).hexdigest()))
    gif=animate(levels,slots)
    (ROOT/"report.json").write_text(json.dumps(dict(edges=edges,return_costs=costs,iterations=iteration,gif=gif),indent=2)+"\n")
    print(edges,gif,flush=True)

if __name__=="__main__":main()
