"""Film a cyclic mosaic with three strictly outward camera moves."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"spider_iron_captain"
REVEAL=ROOT.parent/"dualscale_reveal"
sys.path.insert(0,str(PREV))
from render import return_cell, xy
from sdxl.hero_cycle_render import BOX,LANCZOS,SIDE,project,read,digest

HEROES=("spider_man","iron_man","captain_america")
GRID=8

def inputs():
    gen=read(ROOT/"generation.json")
    stages={s["subject"]:s for s in gen["new_stages"]}
    iron=read(REVEAL/"generation.json")
    stages["iron_man"]=dict(leaves="spider_man",tiles=iron["tiles"])
    assert gen["grid"]==iron["protocol"]["grid"]==GRID
    boards=[];targets=[]
    for j,h in enumerate(HEROES):
        stage=stages[h]
        assert stage["leaves"]==HEROES[(j-1)%3]
        is_iron=h=="iron_man"
        folder=REVEAL if is_iron else ROOT
        src=ROOT.parent/"hero_cycle/opening_target.png" if is_iron else PREV/stage["source"]
        target_sha=iron["protocol"]["target_sha256"] if is_iron else stage["source_png"]
        assert digest(src)==target_sha
        with Image.open(src) as im:targets.append(im.convert("RGB"))
        board=Image.new("RGB",(8192,8192))
        assert [t["index"] for t in stage["tiles"]]==list(range(64))
        for t in stage["tiles"]:
            path=folder/t["file"];assert digest(path)==t["sha256"]
            with Image.open(path) as im:
                assert im.size==(1024,1024)
                board.paste(im,xy(t["index"],1024))
        boards.append(board)
    return boards,targets

def resolve(boards,targets):
    templates=[b.resize((1024,1024),BOX) for b in boards]
    slots=[];costs=[]
    for j in range(3):
        # Every parent is made of the hero the audience saw immediately before it.
        slot,scores=return_cell(templates[j],templates[(j-1)%3],targets[j])
        slots.append(slot);costs.append(scores)
    views=templates
    for _ in range(16):
        nxt=[im.copy() for im in templates]
        for j,i in enumerate(slots):
            nxt[j].paste(views[(j-1)%3].resize((128,128),BOX),xy(i,128))
        stable=all(a.tobytes()==b.tobytes() for a,b in zip(nxt,views))
        views=nxt
        if stable:break
    else:raise RuntimeError("Cyclic previews failed to converge")
    levels=[];edges=[]
    for j,(slot,b) in enumerate(zip(slots,boards)):
        inner=(j-1)%3;x,y=xy(slot,1024);child=views[inner]
        b.paste(child,(x,y))
        assert b.crop((x,y,x+1024,y+1024)).tobytes()==child.tobytes()
        assert b.resize((1024,1024),BOX).tobytes()==views[j].tobytes()
        views[j].save(ROOT/(HEROES[j]+"_overview.png"))
        levels.append([b.reduce(k) if k>1 else b for k in (8,4,2,1)])
        edges.append(dict(outer=HEROES[j],inner=HEROES[inner],row=slot//8+1,column=slot%8+1,
                          inner_pixel_sha256=hashlib.sha256(child.tobytes()).hexdigest()))
    return levels,slots,costs,edges

def field_width(t):
    ease=t*t*t*(10-15*t+6*t*t)
    return GRID**(ease-1)

def project_layers(levels,slots,outer,bounds):
    image=Image.new("RGB",(SIDE,SIDE))
    x,y,side=0.,0.,1.
    for _ in range(4):
        project(image,levels[outer],bounds,x,y,side)
        side/=GRID
        dx,dy=xy(slots[outer],side)
        x+=dx;y+=dy
        outer=(outer-1)%3
    return image

def film(levels,slots):
    # The camera width in parent coordinates increases from one cell to all eight.
    widths=[field_width(k/91) for k in range(92)]
    assert widths[0]==1/GRID and widths[-1]==1 and all(a<b for a,b in zip(widths,widths[1:]))
    holds=[project_layers(levels,slots,j,(0.,0.,1.)) for j in range(3)]
    def frame(inner,t):
        outer=(inner+1)%3
        if t==0:return holds[inner]
        if t==1:return holds[outer]
        width=field_width(t)
        x,y=xy(slots[outer],1/GRID)
        mix=(1-width)/(1-1/GRID)
        return project_layers(levels,slots,outer,(x*mix,y*mix,width))
    atlas=Image.new("RGB",(8*100,9*100))
    for j in range(3):
        for k in range(24):
            idx=j*24+k
            atlas.paste(frame(j,k/23).resize((100,100),LANCZOS),(idx%8*100,idx//8*100))
    palette=atlas.quantize(256,method=Image.Quantize.MEDIANCUT)
    def quantize(im):return im.quantize(palette=palette,dither=Image.Dither.FLOYDSTEINBERG)
    stops=[quantize(im) for im in holds];frames=[];times=[]
    for j in range(3):
        next_j=(j+1)%3;frames.append(stops[j]);times.append(5500)
        movement=[quantize(frame(j,k/91)) for k in range(1,91)]
        while movement and movement[0].tobytes()==stops[j].tobytes():movement.pop(0)
        while movement and movement[-1].tobytes()==stops[next_j].tobytes():movement.pop()
        assert movement
        frames.extend(movement)
        times.extend((10*np.diff(np.rint(np.linspace(0,450,len(movement)+1)).astype(int))).tolist())
    frames.append(quantize(frame(2,1)));times.append(20)
    path=ROOT/"spider_iron_captain_zoom_out.gif"
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=times,loop=0,optimize=False,disposal=1)
    elapsed=0;observed=[];timings=[]
    with Image.open(path) as im:
        assert im.info["loop"]==0
        for k in range(im.n_frames):
            im.seek(k);dt=im.info["duration"]
            if dt>=1000:
                observed.append(im.convert("RGB").tobytes());timings.append((elapsed,dt))
            elapsed+=dt
        assert im.convert("RGB").tobytes()==observed[0]
        count=im.n_frames
    assert observed==[s.convert("RGB").tobytes() for s in stops]
    assert timings==[(0,5500),(10000,5500),(20000,5500)] and elapsed==30020
    # Save actual camera samples: beginning, halfway out, and wide for each reveal.
    strip=Image.new("RGB",(3*240,3*240))
    for j in range(3):
        for k,t in enumerate((0,.5,1)):
            strip.paste(frame(j,t).resize((240,240),LANCZOS),(k*240,j*240))
    strip.save(ROOT/"zoom_sequence.jpg",quality=93)
    return dict(order=[*HEROES,HEROES[0]],camera="outward",width_per_leg=[1/GRID,1.0],
        frames=count,duration_ms=elapsed,holds=timings,first_last_identical=True,gif_sha256=digest(path))

def main():
    boards,targets=inputs()
    levels,slots,costs,edges=resolve(boards,targets)
    gif=film(levels,slots)
    (ROOT/"report.json").write_text(json.dumps(dict(gif=gif,edges=edges,slot_costs=costs),indent=2)+"\n")
    print(gif,flush=True)
if __name__=="__main__":main()
