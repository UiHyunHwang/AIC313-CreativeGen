"""Validate one child, render its zoom sequence, or package both in an offline viewer."""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from select_tile import rank_tiles, model_id, revision

root=Path(__file__).resolve().parent
selection=json.loads((root/"selection.json").read_text())
parent=root.parents[1]/selection["parent"]
selected=selection["ranked"][0]["tile"]
def coord(name):
    a,b=name.split("_")
    return int(a[1:]),int(b[1:])
row,col=coord(selected)
L=Image.Resampling.LANCZOS
def read_json(path):
    return json.loads(path.read_text())
def run_dir(n):
    runs=[p.parent for p in (root/f"inner_{n}").glob("sdxl_mosaic_*/manifest.json") if read_json(p)["status"]=="complete"]
    if len(runs)!=1:
        raise ValueError(f"Expected one completed run for inner {n}, found {runs}")
    return runs[0]
def rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")
def jpeg_data(im,q):
    buffer=io.BytesIO()
    im.save(buffer,"JPEG",quality=q)
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def viewport(board, rect, box, side):
    # Project only the visible intersection; this also bounds memory at maximum zoom.
    left,top,right,bottom=rect
    x,y,w=box
    x0,y0=max(x,left),max(y,top)
    x1,y1=min(x+w,right),min(y+w,bottom)
    if x1<=x0 or y1<=y0:
        return
    scale=side/(right-left)
    dst=tuple(round(v*scale) for v in (x0-left,y0-top,x1-left,y1-top))
    if dst[2]<=dst[0] or dst[3]<=dst[1]:
        return
    k=board.width/w
    src=tuple(max(0,min(board.width,v*k)) for v in (x0-x,y0-y,x1-x,y1-y))
    return dst,board.resize((dst[2]-dst[0],dst[3]-dst[1]),L,box=src)

def make_zoom(n,outer,inner,example,ir,ic,path):
    side=512
    frames=[]
    def draw(z):
        depth=math.log2(z)
        mix=max(0,min(1,(depth-4)/math.log2(n)))
        cx=(col+.5+mix*((ic+.5)/n-.5))/16
        cy=(row+.5+mix*((ir+.5)/n-.5))/16
        half=.5/z
        cx=max(half,min(1-half,cx))
        cy=max(half,min(1-half,cy))
        rect=(cx-half,cy-half,cx+half,cy+half)
        frame=Image.new("RGB",(side,side))
        for board,box in [(outer,(0,0,1)),(inner,(col/16,row/16,1/16)),(example,((col+ic/n)/16,(row+ir/n)/16,1/(16*n)))]:
            part=viewport(board,rect,box,side)
            if part:
                dst,patch=part
                frame.paste(patch,dst[:2])
        frames.append(frame)
    times=[]
    for start,end,count,hold in ((0,4,31,1450),(4,4+math.log2(n),31,1750)):
        for i in range(0 if not frames else 1,count):
            f=i/(count-1)
            draw(2**(start+(end-start)*f*f*(3-2*f)))
            times.append(hold if i==count-1 else (1200 if len(times)==0 else 75))
    frames[0].save(path,"WEBP",save_all=True,append_images=frames[1:],duration=times,quality=77,method=3,loop=0)
    with Image.open(path) as check:
        assert check.n_frames>=55 and check.size==(side,side)

def finish(n):
    path=run_dir(n)
    manifest=read_json(path/"manifest.json")
    guard=read_json(path/"memory_guard.json") if (path/"memory_guard.json").is_file() else None
    cfg=read_json(path/"config.json")
    assert cfg["tile_prompt"]=="captain america" and cfg["noise_steps"]==12
    assert (cfg["parent_tile_source"]["row"],cfg["parent_tile_source"]["column"])==(row,col)
    assert (guard is None and cfg["min_free_gpu_gib"] == 0) or guard["status"]=="monitoring_finished"
    assert manifest["derived"]["tile_start_timestep"]==836
    tiles=sorted((path/"tiles").glob("*.png"))
    assert len(tiles)==n*n
    checked=[(path/"target.png",1024),(path/"mosaic.png",1024*n)]
    for j,p in enumerate(tiles):
        assert coord(p.stem)==divmod(j,n)
        checked.append((p,1024))
    for p,sz in checked:
        with Image.open(p) as im:
            assert im.format=="PNG" and im.size==(sz,sz)
            im.verify()
        with p.open("rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest()==manifest["outputs"][str(p.relative_to(path))]["sha256"]
    positive=["Captain America with his round star shield and a blue helmet","a clear portrait of Marvel superhero Captain America, red white and blue shield"]
    negative=["an abstract patch of texture","a tiny figure in an empty landscape","Spider-Man with a red webbed mask","a flag without a superhero"]
    ranks=rank_tiles(tiles,positive,negative)
    ir,ic=coord(ranks[0]["tile"])
    (path/"example_selection.json").write_text(json.dumps({"positive":positive,"negative":negative,"ranked":ranks,"model":model_id,"revision":revision},indent=2))
    exemplar=rgb(tiles[ir*n+ic])
    with Image.open(path/"mosaic.png") as full:
        inner=full.convert("RGB").resize((3072,3072),L)
    inner.resize((1024,1024),L).save(path/"preview.jpg",quality=94)
    with Image.open(parent/"mosaic.png") as full:
        outer=full.convert("RGB").resize((3072,3072),L)
    outer.paste(inner.resize((192,192),L),(col*192,row*192))
    outer.resize((1536,1536),L).save(path/"outer_overview.jpg",quality=94)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
    sheet=Image.new("RGB",(1440,418),"white")
    pen=ImageDraw.Draw(sheet)
    for i,(label,img) in enumerate([("Iron Man | outer 16 x 16",outer),("Spider-Man | selected tile",rgb(parent/"tiles"/(selected+".png"))),(f"Spider-Man | inner {n} x {n}",inner),("Captain America | detail",exemplar)]):
        pen.text((i*360+5,8),label,font=font,fill="#221e1c")
        sheet.paste(img.resize((350,350),L),(i*360+5,45))
    sheet.save(path/"levels.jpg",quality=94)
    make_zoom(n,outer,inner,exemplar,ir,ic,path/"zoom.webp")
    (path/"provenance.json").write_text(json.dumps({"parent":selection["parent"],"parent_config_sha256":hashlib.sha256((parent/"config.json").read_bytes()).hexdigest(),"parent_selection":"../../selection.json","parent_candidates":256,"parent_tile":selected,"inner_prompt":"captain america","inner_tile_count":n*n,"outer_prompt_provenance":read_json(parent/"prompt_provenance.json"),"inner_example":ranks[0]["tile"],"nested_previews":"The chosen parent cell displays the reduced child mosaic at all zoom levels. Preview and zoom files are classical resampling of the generated layers."},indent=2)+"\n")
    print(n,path,"example",ranks[0],manifest["total_seconds"],manifest["memory"],guard and guard["minimum_observed_free_gib"],flush=True)

def package():
    def layer(path,n,is_parent):
        with Image.open(path/"mosaic.png") as im:
            preview=jpeg_data(im.resize((1800,1800),L),90)
        tiles=[]
        for p in sorted((path/"tiles").glob("*.png")):
            with Image.open(p) as im:
                im=im.convert("RGB")
                if is_parent:
                    im=im.resize((512,512),L)
                tiles.append(jpeg_data(im,82))
        assert len(tiles)==n*n
        return {"preview":preview,"grid":n,"tiles":tiles}
    assets={"selected":[row,col],"outer":layer(parent,16,True),"inner":{}}
    for n in [8,16]:
        run=run_dir(n)
        assets["inner"][n]=layer(run,n,False)
        assets["inner"][n]["example"]=coord(read_json(run/"example_selection.json")["ranked"][0]["tile"])
    assert (run_dir(8)/"target.png").read_bytes()==(run_dir(16)/"target.png").read_bytes()
    page=(root/"viewer_template.html").read_text()
    (root/"zoom.html").write_text(page.replace("__ASSETS__",json.dumps(assets,separators=(",",":"))))
    print("Viewer ready", (root/"zoom.html").stat().st_size,flush=True)

if __name__=="__main__":
    Image.MAX_IMAGE_PIXELS=16384**2
    parser=argparse.ArgumentParser()
    parser.add_argument("part",choices=["8","16","viewer"])
    part=parser.parse_args().part
    package() if part=="viewer" else finish(int(part))
