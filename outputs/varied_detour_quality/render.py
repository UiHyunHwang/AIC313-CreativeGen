"""Render a four-board route with two independent links on the Captain America grid."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from PIL import Image,ImageDraw,ImageChops
from quality import apply_quality
from sdxl.hero_cycle_render import BOX,LANCZOS,SIDE,cell,digest,project,read,xy

def pixel_sha(im):
    return hashlib.sha256(im.tobytes()).hexdigest()

def portals(base,selection):
    boards=read(base/"report.json")["boards"]
    first,second=[(d["row_one_based"]-1)*16+d["column_one_based"]-1 for d in boards[:2]]
    detour=selection["hero_index"];back=selection["chosen"]["index"]
    assert len({detour,back,selection["old"]["index"]})==3
    return [{first:1},{second:2},{detour:3,back:0},{}]

def build(root,runs,edges,selection,overrides):
    templates=[]
    for run in runs:
        with Image.open(run/"mosaic.png") as im:
            assert im.size==(16384,16384)
            templates.append(im.resize((1024,1024),BOX))
    for j,replacements in enumerate(overrides):
        for index,path in replacements.items():
            with Image.open(path) as tile:templates[j].paste(tile.resize((64,64),BOX),xy(index,64))
    thumbs=templates.copy()
    for _ in range(16):
        nxt=[im.copy() for im in templates]
        for j,mapping in enumerate(edges):
            for k,child in mapping.items():nxt[j].paste(thumbs[child].resize((64,64),BOX),xy(k,64))
        stable=all(a.tobytes()==b.tobytes() for a,b in zip(thumbs,nxt))
        thumbs=nxt
        if stable: break
    assert stable
    pyramids=[];proof=[]
    for j,run in enumerate(runs):
        with Image.open(run/"mosaic.png") as im:board=im.convert("RGB")
        for index,path in overrides[j].items():
            with Image.open(path) as tile:board.paste(tile,xy(index,1024))
        original_tile=cell(board,selection["previous_choice"]["index"]) if j==2 else None
        for k,child in edges[j].items():
            board.paste(thumbs[child],xy(k,1024))
            checksum=pixel_sha(cell(board,k))
            assert checksum==pixel_sha(thumbs[child])
            proof.append(dict(parent=j,child=child,row=k//16+1,column=k%16+1,sha256=checksum))
        if original_tile is not None: assert original_tile.tobytes()==cell(board,selection["previous_choice"]["index"]).tobytes()
        assert board.resize((1024,1024),BOX).tobytes()==thumbs[j].tobytes()
        thumbs[j].save(root/f"board_{j}_overview.png")
        if j==2:board.save(root/"captain_two_destinations.png",compress_level=2)
        pyramids.append([board.reduce(k) for k in (16,8,4,2)])
        del board
    # Compare placements on the same overview, with the additional excursion tile held fixed.
    placement_comparison(root,templates[2],thumbs[3],thumbs[0],selection)
    return pyramids,proof

def placement_comparison(root,template,detail,opening,selection):
    sel=selection["chosen"]["index"]
    variants=list(dict.fromkeys([selection["previous_choice"]["index"],sel]))
    strip=Image.new("RGB",(384*len(variants),640),"white"); pen=ImageDraw.Draw(strip)
    for j,k in enumerate(variants):
        sample=template.copy()
        sample.paste(detail.resize((64,64),BOX),xy(selection["hero_index"],64))
        sample.paste(opening.resize((64,64),BOX),xy(k,64))
        strip.paste(sample.resize((384,384),LANCZOS),(384*j,30))
        x,y=xy(k,64)
        local=sample.crop((x-64,y-64,x+128,y+128)).resize((208,208),LANCZOS)
        strip.paste(local,(384*j+88,422))
        label="Selected" if k==sel else "Previous" if k==selection["previous_choice"]["index"] else "Suggested"
        pen.text((384*j+7,7),f"{label}  row {k//16+1}, col {k%16+1}",fill="black")
    strip.save(root/"placements.jpg",quality=94)

def view(levels,edges,parent,slot,progress):
    if progress==1:
        parent=edges[parent][slot]; progress=0
        slot=0
    f=progress
    width=16**(-(f**3*(10-15*f+6*f*f)))
    shift=(1-width)/(1-1/16)
    bx,by=xy(slot,1/16)
    bounds=(bx*shift,by*shift,width)
    im=Image.new("RGB",(SIDE,SIDE))
    project(im,levels[parent],bounds,0,0,1)
    for index,child in edges[parent].items():
        x,y=xy(index,1/16)
        project(im,levels[child],bounds,x,y,1/16)
    return im

def animate(root,levels,edges,sel):
    outer=next(iter(edges[0]));spider=next(iter(edges[1]))
    hero,back=sel["hero_index"],sel["chosen"]["index"]
    legs=[(0,outer,False),(1,spider,False),(2,hero,False),(2,hero,True),(2,back,False)]
    poses=[(0,outer,0,5500)]
    holds=[0]
    for j,(parent,slot,reverse) in enumerate(legs):
        for step in range(1,91):
            f=step/91
            poses.append((parent,slot,1-f if reverse else f,50))
        holds.append(len(poses))
        poses.append((parent,slot,0 if reverse else 1,5500 if j<4 else 20))
    def frame(p):return view(levels,edges,*p[:3])
    samples=poses[::5]+[poses[i] for i in holds]
    atlas=Image.new("RGB",(8*100,math.ceil(len(samples)/8)*100))
    for i,p in enumerate(samples):atlas.paste(frame(p).resize((100,100),LANCZOS),((i%8)*100,i//8*100))
    palette=atlas.quantize(256,method=Image.Quantize.MEDIANCUT)
    frames=[frame(p).quantize(palette=palette,dither=Image.Dither.FLOYDSTEINBERG) for p in poses]
    assert frames[0].tobytes()==frames[-1].tobytes()
    # GIF coalesces equal frames. Keep rounding at extreme zoom from turning
    # a 5.5s hold into a longer hold when the camera reverses in the same cell.
    compact_frames=[frames[0]]
    compact_poses=[poses[0]]
    compact_holds=[0]
    for a,b in zip(holds,holds[1:]):
        start,end=frames[a].convert("RGB"),frames[b].convert("RGB")
        moving=list(range(a+1,b))
        while moving and ImageChops.difference(start,frames[moving[0]].convert("RGB")).getbbox() is None:
            moving.pop(0)
        while moving and ImageChops.difference(end,frames[moving[-1]].convert("RGB")).getbbox() is None:
            moving.pop()
        assert moving
        ticks=np.rint(np.linspace(0,450,len(moving)+1)).astype(int)
        for idx,dt in zip(moving,np.diff(ticks)):
            compact_frames.append(frames[idx])
            compact_poses.append((*poses[idx][:3],10*int(dt)))
        compact_holds.append(len(compact_poses))
        compact_frames.append(frames[b])
        compact_poses.append(poses[b])
    frames,poses,holds=compact_frames,compact_poses,compact_holds
    gif=root/"ironman_detour.gif"
    frames[0].save(gif,save_all=True,append_images=frames[1:],optimize=False,loop=0,disposal=1,duration=[p[-1] for p in poses])
    halt_at=[sum(p[-1] for p in poses[:i]) for i in holds]
    captured=[];pause_lengths=[];ms=0
    with Image.open(gif) as decoded:
        assert decoded.info["loop"]==0
        for i in range(decoded.n_frames):
            decoded.seek(i);dur=decoded.info["duration"]
            if dur>=1000:pause_lengths.append(dur)
            for t in halt_at:
                if ms<=t<ms+dur:captured.append(decoded.convert("RGB"))
            ms+=dur
        count=decoded.n_frames
    assert len(captured)==6 and len(pause_lengths)==5 and all(5500<=d<=5600 for d in pause_lengths)
    assert ms==50020 and captured[0].tobytes()==captured[-1].tobytes() and captured[2].tobytes()==captured[4].tobytes()
    for im,i in zip(captured,holds):
        assert np.abs(np.asarray(im,dtype=float)-np.asarray(frame(poses[i]),dtype=float)).mean()<12
    return dict(duration_ms=ms,frames=count,pauses_ms=pause_lengths,first_last_identical=True,captain_before_after_identical=True,sha256=digest(gif))

def main(root,base):
    sel=read(root/"selection.json")
    parents=[(base/p).resolve() for p in read(base/"report.json")["raw_runs"]]
    done=[p.parent for p in (root/"new_board").glob("*/manifest.json") if read(p)["status"]=="complete"]
    assert len(done)==1
    cfg=read(done[0]/"config.json")
    source=cfg["parent_tile_source"];i=sel["hero_index"]
    assert source["sha256"]==digest(parents[2]/"final_latents.safetensors") and (source["row"],source["column"])==divmod(i,16)
    assert cfg["tile_prompt"]=="iron man" and cfg["height"]==cfg["width"]==16384
    leaves=list((done[0]/"tiles").glob("*.png"));assert len(leaves)==256
    manifest=read(done[0]/"manifest.json")
    assert all(digest(p)==manifest["outputs"]["tiles/"+p.name]["sha256"] for p in leaves)
    edges=portals(base,sel)
    overrides=apply_quality(root,base,parents)
    levels,checks=build(root,parents+done,edges,sel,overrides)
    video=animate(root,levels,edges,sel)
    (root/"report.json").write_text(json.dumps(dict(gif=video,new_board=str(done[0]),new_sdxl_tile_count=256,edges=checks,
        generation_seconds=manifest["total_seconds"],old_slot_restored=True),indent=2))
    with ZipFile(base/"reproduction.zip") as orig,ZipFile(root/"reproduction.zip","w",compression=ZIP_DEFLATED) as out:
        for name in orig.namelist():out.writestr(name,orig.read(name))
        for name in ["compare.py","review_quality.py","quality.py","render.py","select_route.py","README.md","third_recipe.py","rank_zoom.py"]:out.write(Path(__file__).resolve().parent/name,"detour/"+name)
        for name in ["recipe.json","selection.json","report.json","quality.json","spider_review.json"]:out.write(root/name,"detour/recorded/"+name)
        for data in ("hero_cycle","captain_refined"):
            out.write(root.parent/data/"reproduction.zip","prerequisites/"+data+".zip")
    with ZipFile(root/"reproduction.zip") as bundle:assert bundle.testzip() is None
    print("Verified GIF",video,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--base",type=Path,default=Path("outputs/hero_cycle_varied"))
    args=p.parse_args();main(args.output,args.base)
