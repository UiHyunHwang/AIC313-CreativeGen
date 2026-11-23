"""Rasterize the three-node recursive tile grid and verify a seamless looping GIF."""
import argparse
import hashlib
import json
import math
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

import numpy as np
from PIL import Image

NODES = ("spider", "captain", "iron")  # Node names identify the subjects of their small tiles.
SIDE = 704
BOX = Image.Resampling.BOX
LANCZOS = Image.Resampling.LANCZOS
Image.MAX_IMAGE_PIXELS = 16384**2


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def xy(index, side):
    return index % 16 * side, index // 16 * side


def cell(image, index):
    size=image.width//16
    x,y=xy(index,size)
    return image.crop((x,y,x+size,y+size))


def texture(root, record):
    return (root/record["file"]).with_suffix(".png")


def baseline_preview(root, name):
    folder = root / name
    choice = read(folder/"pilot.json")["best"]
    records = [read(path) for path in (folder/"samples").glob("*_s43.json")]
    records = [r for r in records if r["prompt"] == choice["prompt"] and r["steps"] == choice["steps"]]
    assert {r["index"] for r in records} == set(range(256))
    canvas = Image.new("RGB", (1024,1024))
    for record in records:
        with Image.open(texture(root,record)) as image:
            canvas.paste(image.resize((64,64),BOX),xy(record["index"],64))
    canvas.save(folder/"before_selection.png")


def resolve(root, results, links):
    # One of 256 disjoint cells per node points to another node. The rest are leaf samples.
    # Resolve at the tile's native viewing size until the cyclic reference has a pixel-stable value.
    templates=[]
    for node,link in zip(results,links):
        base=Image.new("RGB",(1024,1024))
        for rec in node["tiles"]:
            if rec["index"]==link:
                continue
            with Image.open(texture(root,rec)) as im:
                base.paste(im.resize((64,64),BOX),xy(rec["index"],64))
        templates.append(base)
    thumbs=[im.copy() for im in templates]
    for iteration in range(1,17):
        update=[im.copy() for im in templates]
        for j,im in enumerate(update):
            im.paste(thumbs[(j+1)%3].resize((64,64),BOX),xy(links[j],64))
        same=all(a.tobytes()==b.tobytes() for a,b in zip(thumbs,update))
        thumbs=update
        if same:
            break
    else:
        raise RuntimeError("Cyclic BOX downsampling did not converge to stable thumbnail pixels")
    return thumbs,iteration


def write_boards(root,results,links,thumbs):
    pyramids=[]
    checks=[]
    for j,(name,records,link) in enumerate(zip(NODES,results,links)):
        full=Image.new("RGB",(16384,16384))
        leaves={r["index"]:r for r in records["tiles"] if r["index"]!=link}
        assert set(leaves)==set(range(256))-{link}
        for i in range(256):
            if i==link:
                full.paste(thumbs[(j+1)%3],xy(i,1024))
            else:
                with Image.open(texture(root,leaves[i])) as tile:
                    assert tile.size==(1024,1024)
                    full.paste(tile,xy(i,1024))
        assert full.resize((1024,1024),BOX).tobytes()==thumbs[j].tobytes()
        assert cell(full,link).tobytes()==thumbs[(j+1)%3].tobytes()
        path=root/name/"mosaic.png"
        full.save(path,compress_level=3)
        thumbs[j].save(root/name/"overview.png")
        checks.append(dict(node=name,linked_node=NODES[(j+1)%3],row_one_based=link//16+1,column_one_based=link%16+1,
            mosaic_sha256=digest(path),overview_sha256=digest(root/name/"overview.png"),
            outgoing_cell_pixel_sha256=hashlib.sha256(cell(full,link).tobytes()).hexdigest(),
            overview_pixel_sha256=hashlib.sha256(thumbs[j].tobytes()).hexdigest()))
        # 704px × 16 zoom never exceeds 11264 source pixels. Keep full resolution on disk;
        # the final part of each leg samples the child directly at all zoom levels.
        pyramids.append([full.reduce(k) for k in (16,8,4,2)])
        del full
    assert all(d["outgoing_cell_pixel_sha256"]==checks[(i+1)%3]["overview_pixel_sha256"] for i,d in enumerate(checks))
    return pyramids,checks


def project(frame,levels,bounds,bx,by,bw):
    x,y,w=bounds
    x0,y0=max(x,bx),max(y,by)
    x1,y1=min(x+w,bx+bw),min(y+w,by+bw)
    if x1<=x0 or y1<=y0: return
    dest=tuple(round(val*SIDE/w) for val in (x0-x,y0-y,x1-x,y1-y))
    if dest[2]<=dest[0] or dest[3]<=dest[1]: return
    desired=SIDE*bw/w
    source=next((im for im in levels if im.width>=desired),levels[-1])
    src=tuple(max(0,min(source.width,val*source.width/bw)) for val in (x0-bx,y0-by,x1-bx,y1-by))
    tile=source.resize((dest[2]-dest[0],dest[3]-dest[1]),LANCZOS,box=src)
    frame.paste(tile,dest[:2])


def draw(pyramids,links,j,f):
    if f==1:
        j=(j+1)%3
        f=0
    ease=f**3*(10-15*f+6*f*f)
    w=16**(-ease)
    a=(1-w)/(1-1/16)
    bx,by=xy(links[j],1/16)
    bounds=a*bx,a*by,w
    frame=Image.new("RGB",(SIDE,SIDE))
    project(frame,pyramids[j],bounds,0,0,1)
    project(frame,pyramids[(j+1)%3],bounds,bx,by,1/16)
    return frame


def movie(root,pyramids,links):
    samples=[(0,0.,1050)]
    for node in range(3):
        for i in range(1,73):
            samples.append((node,i/72,50 if i<72 else (1250 if node<2 else 650)))
    # Share a single palette across the cycle, including both copies of the opening image.
    strips=[draw(pyramids,links,j,f).resize((120,120),LANCZOS) for j,f,_ in samples[::4]]
    atlas=Image.new("RGB",(120*8,120*math.ceil(len(strips)/8)))
    for k,im in enumerate(strips):
        atlas.paste(im,(120*(k%8),120*(k//8)))
    palette=atlas.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    frames=[draw(pyramids,links,j,f).quantize(palette=palette,dither=Image.Dither.FLOYDSTEINBERG) for j,f,_ in samples]
    assert frames[0].tobytes()==frames[-1].tobytes()
    path=root/"ironman_cycle.gif"
    frames[0].save(path,save_all=True,append_images=frames[1:],loop=0,disposal=1,optimize=False,
                   duration=[duration for _,_,duration in samples])
    duration=0
    with Image.open(path) as check:
        assert check.info["loop"]==0 and check.size==(SIDE,SIDE)
        initial=check.convert("RGB").tobytes()
        endpoints={}
        times=[0]+[sum(s[2] for s in samples[:k]) for k in (72,144,216)]
        for i in range(check.n_frames):
            check.seek(i)
            frame=check.convert("RGB")
            end=duration+check.info["duration"]
            for k,t in enumerate(times):
                if duration<=t<end:
                    ref=draw(pyramids,links,k%3,0.)
                    endpoints[k]=float(np.abs(np.asarray(frame,dtype=np.float32)-np.asarray(ref,dtype=np.float32)).mean())
            duration=end
        assert frame.tobytes()==initial and check.n_frames>=180
        assert set(endpoints)=={0,1,2,3} and max(endpoints.values())<12
        assert duration==sum(s[2] for s in samples)
        return dict(file=path.name,sha256=digest(path),width=SIDE,frames=check.n_frames,
                    duration_ms=duration,loop=0,first_last_rgb_identical=True,stop_rgb_mae_255=endpoints)


def summary(root):
    results=[read(root/node/"selected.json") for node in NODES]
    links=[results[0]["best"]["index"],results[1]["best"]["index"],results[2]["reserved"]]
    for name in NODES[:2]:
        baseline_preview(root, name)
    thumbs,iterations=resolve(root,results,links)
    pyramids,checks=write_boards(root,results,links,thumbs)
    video=movie(root,pyramids,links)
    stats={}
    for name,node in zip(NODES,results):
        level=root/name
        samples=list((level/"samples").glob("*.json"))
        pilot=read(level/"pilot.json") if (level/"pilot.json").exists() else {}
        measures={}
        for key in ("base","tiles"):
            measures[key]={field:dict(mean=float(np.mean(vals)),q10=float(np.quantile(vals,.10)))
                         for field in ("score","clip","rgb_mae") if (vals:=[r[field] for r in node[key]])}
        stats[name]=dict(sample_count=len(samples),standalone_controls=len(pilot.get("controls",[])),pilot=pilot,
                unique_base_coordinates=len(node["base"]),rerolls=sum(read(p)["seed"]!=43 for p in samples),
                changed_by_selection=sum(a["seed"]!=b["seed"] for a,b in zip(node["base"],node["tiles"])),metrics=measures)
    for j, name in enumerate(NODES):
        target = (root/"opening_target.png" if j == 0 else texture(root, results[j-1]["best"]))
        with Image.open(target) as reference:
            original = np.asarray(reference.convert("RGB").resize((32,32), BOX), dtype=np.float32)
        measured = np.asarray(thumbs[j].resize((32,32), BOX), dtype=np.float32)
        stats[name]["macro_rgb_mae_32_on_0_to_1"] = float(np.abs(original-measured).mean()/255)
    report=dict(gif=video,nodes=checks,thumbnail_iterations=iterations,stats=stats,
              total_generated_samples=1+sum(r["sample_count"]+r["standalone_controls"] for r in stats.values()),
              count_note="One opening bootstrap; independent tile candidates (pilot/final overlap deduplicated) and controls. Cyclic references and camera frames are not new SDXL samples.")
    (root/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    repo = Path(__file__).resolve().parents[1]
    source_files = [repo/"LICENSE", repo/"sdxl/hero_cycle_requirements.txt", repo/"tests/test_sdxl_hero_cycle.py"]
    source_files.extend(sorted((repo/"sdxl").glob("*.py")))
    source_files.extend([repo/"sdxl/HERO_CYCLE.md", repo/"sdxl/hero_cycle_recipe.json"])
    with ZipFile(root/"reproduction.zip","w",compression=ZIP_DEFLATED) as archive:
        for item in source_files:
            archive.write(item,str(item.relative_to(repo)))
        for item in ["environment.json","protocol.json","report.json"]:
            archive.write(root/item,"recorded/"+item)
    print("Verified",video,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("outputs/hero_cycle"))
    summary(p.parse_args().output)
