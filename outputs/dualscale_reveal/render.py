"""Render one continuous camera move over a single SDXL board."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
SIDE=704
LANCZOS=Image.Resampling.LANCZOS
BOX=Image.Resampling.BOX

def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def compose(rows,grid):
    board=Image.new("RGB",(grid*1024,grid*1024))
    assert sorted(r["index"] for r in rows)==list(range(grid*grid))
    for r in rows:
        path=ROOT/r["file"]
        assert digest(path)==r["sha256"]
        with Image.open(path) as im:
            assert im.size==(1024,1024)
            x,y=(r["column"]-1)*1024,(r["row"]-1)*1024
            board.paste(im,(x,y))
    return board

def render():
    data=json.loads((ROOT/"generation.json").read_text())
    rows=data["tiles"];grid=data["protocol"]["grid"]
    board=compose(rows,grid)
    board.save(ROOT/"mosaic.png",compress_level=2)
    overview=board.resize((SIDE,SIDE),LANCZOS)
    overview.save(ROOT/"overview.png")
    start=max(rows,key=lambda r:(min(r["native"],r["display44"]),-r["index"]))
    x0,y0=(start["column"]-1)*1024,(start["row"]-1)*1024
    pyramid=[board.reduce(2**k) for k in (3,2,1)]+[board]
    def frame(progress):
        if progress==1:return overview
        # Smoothly open the field of view while centering on the full square.
        f=progress**3*(10-15*progress+6*progress**2)
        w=grid**f
        t=(w-1)/(grid-1)
        x,y= (1-t)*x0, (1-t)*y0
        desired=SIDE*grid/w
        src=next((im for im in pyramid if im.width>=desired),board)
        scale=src.width/board.width
        bounds=(x*scale,y*scale,min(src.width,(x+1024*w)*scale),min(src.height,(y+1024*w)*scale))
        return src.resize((SIDE,SIDE),LANCZOS,box=bounds)
    # Derive a shared palette from the actual movie to avoid flicker.
    atlas=Image.new("RGB",(6*110,5*110))
    for i in range(30):atlas.paste(frame(i/29).resize((110,110),LANCZOS),((i%6)*110,(i//6)*110))
    palette=atlas.quantize(256,method=Image.Quantize.MEDIANCUT)
    frames=[frame(i/100).quantize(palette=palette,dither=Image.Dither.FLOYDSTEINBERG) for i in range(101)]
    first,last=frames[0],frames[-1]
    inner=frames[1:-1]
    while inner and inner[0].tobytes()==first.tobytes():inner.pop(0)
    while inner and inner[-1].tobytes()==last.tobytes():inner.pop()
    assert inner
    film=[first,*inner,last,*reversed(inner),first]
    def times(ticks):
        return (10*np.diff(np.rint(np.linspace(0,ticks,len(inner)+1)).astype(int))).tolist()
    times_ms=[5500,*times(500),5500,*times(450),20]
    gif=ROOT/"spider_to_ironman.gif"
    film[0].save(gif,save_all=True,append_images=film[1:],duration=times_ms,optimize=False,loop=0,disposal=1)
    holds=[];total=0
    with Image.open(gif) as im:
        assert im.info["loop"]==0
        for i in range(im.n_frames):
            im.seek(i);dt=im.info["duration"]
            if dt>=1000:holds.append((total,dt,im.convert("RGB").tobytes()))
            total+=dt
        n=im.n_frames
        assert im.convert("RGB").tobytes()==holds[0][2]
    assert total==20520
    assert [(t,d) for t,d,_ in holds]==[(0,5500),(10500,5500)]
    assert holds[1][2]==last.convert("RGB").tobytes()
    manifest=dict(board_size=board.size,selected_row=start["row"],selected_column=start["column"],
        selection_rule="Maximum of the weaker of native and 44-pixel Spider-Man CLIP margins, all 64 cells",
        frames=n,duration_ms=total,hold_ms=5500,loop=True,verified_first_last_identical=True,
        gif_sha256=digest(gif),board_sha256=digest(ROOT/"mosaic.png"))
    (ROOT/"report.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(manifest,flush=True)

if __name__=="__main__":render()
