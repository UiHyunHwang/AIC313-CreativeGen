"""Render the adopted policy, compare the actual display, and package reproducibility inputs."""
from pathlib import Path
import json
import sys
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from PIL import Image,ImageDraw
from outputs.outward_palette16 import render16,camera,gif_colors,quality,package_run

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/"outward_palette16"
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+"\n")
def compare():
    sheet=Image.new("RGB",(3*320,2*348),"#faf9f5")
    d=ImageDraw.Draw(sheet);data={}
    for j,h in enumerate(render16.HEROES):
        data[h]={}
        for k,(label,folder) in enumerate((("First 16 x 16",BASE),("Scene lighting",ROOT))):
            with Image.open(folder/(h+"_overview.png")) as f:
                im=f.convert("RGB")
            sheet.paste(im.resize((320,320),Image.Resampling.LANCZOS),(320*j,348*k))
            d.text((320*j+5,348*k+323),label+"   "+h,fill="#222")
            a=np.asarray(im.resize((16,16),Image.Resampling.BOX))/255
            yy,xx=np.indices((16,16))
            edge=(xx<3)|(xx>12)|(yy<3)|(yy>12)
            pale=(a@[.2126,.7152,.0722]>.52)&(np.ptp(a,axis=-1)<.16)
            data[h][label]={"outer_cells":int(edge.sum()),"pale_neutral_outer_cells":int((pale&edge).sum())}
    sheet.save(ROOT/"scene_comparison.jpg",quality=96)
    write(ROOT/"perimeter.json",data)
def bundle():
    project=ROOT.parents[1]
    gen=read(ROOT/"generation.json");report=read(ROOT/"render.json")
    assert report["gif"]["duration_ms"]==30020 and report["gif"]["first_last_identical"]
    assert report["gif"]["gif_sha256"]==package_run.sha(ROOT/"spider_iron_captain_zoom_out.gif")
    counts={}
    for st in gen["stages"]:
        assert len(st["tiles"])==len({t["sha256"] for t in st["tiles"]})==256
        assert len({t["seed"] for t in st["tiles"]})==256
        with Image.open(ROOT/(st["subject"]+"_16x16.jpg")) as im:
            assert im.size==(16384,16384);im.verify()
        counts[st["subject"]]=sum(t["active"] for t in st["tiles"])
    write(ROOT/"verified.json",dict(active=counts,unique_per_board=256,grid=16,gif=report["gif"],
         peak_reserved_gib=gen["peak_reserved_gib"],quality=read(ROOT/"quality.json")["stats"],
         display_colors=read(ROOT/"gif_colors.json"),perimeter=read(ROOT/"perimeter.json")))
    target=ROOT/"reproduction.zip"
    with ZipFile(target,"w",ZIP_DEFLATED,compresslevel=3) as z:
        def add(p):z.write(p,p.relative_to(project))
        for p in sorted((project/"sdxl").iterdir()):
            if p.is_file() and p.suffix in (".py",".md",".txt"):add(p)
        for name in ("LICENSE","requirements-sdxl.txt"):add(project/name)
        for folder in (BASE,ROOT):
            for p in sorted(folder.iterdir()):
                if p.is_file() and (p.suffix in (".py",".md",".json") or p.name in ("opening.png","opening.safetensors")):add(p)
            for p in sorted(folder.glob("*/stage.json")):add(p)
        for p in sorted((BASE/"baseline").iterdir()):
            if p.suffix in (".json",".png"):add(p)
        for p in sorted((BASE/"targets").glob("*.json")):add(p)
    with ZipFile(target) as z:assert z.testzip() is None
def main():
    # The shared renderer uses the previous 8x8 gallery as a comparison reference.
    saved_old_results=render16.old_results()
    render16.ROOT=ROOT;render16.old_results=lambda:saved_old_results
    camera.ROOT=ROOT
    render16.main()
    compare()
    gif_colors.ROOT=ROOT;gif_colors.old_results=lambda:saved_old_results
    gif_colors.baseline=lambda:[Image.open(BASE/"baseline"/("gif_"+h+".png")).convert("RGB") for h in render16.HEROES]
    gif_colors.main()
    quality.ROOT=ROOT
    (ROOT/"prior_quality.json").write_bytes((BASE/"prior_quality.json").read_bytes())
    quality.main()
    bundle()
if __name__=="__main__":main()
