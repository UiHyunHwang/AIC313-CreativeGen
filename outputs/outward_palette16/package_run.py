"""Check the completed export and archive compact source and generation records."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from PIL import Image

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
Image.MAX_IMAGE_PIXELS=16384**2
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    gen=read(ROOT/"generation.json");render=read(ROOT/"render.json");quality=read(ROOT/"quality.json")
    targets=read(ROOT/"targets.json")
    files=[];seeds=[]
    for j,stage in enumerate(gen["stages"]):
        hero=stage["subject"]
        assert len(stage["tiles"])==256
        assert len({row["sha256"] for row in stage["tiles"]})==256
        seeds.extend(row["seed"] for row in stage["tiles"])
        assert quality["stats"][hero]["current"]["n"]==255
        assert sha(ROOT.parent/targets["selected"][j])==targets["hashes"][j]
        p=ROOT/(hero+"_16x16.jpg")
        with Image.open(p) as im:
            assert im.size==(16384,16384)
            im.verify()
        files.append(dict(file=p.name,sha256=sha(p),side=16384))
    assert len(set(seeds))==768
    gif=ROOT/"spider_iron_captain_zoom_out.gif"
    assert sha(gif)==render["gif"]["gif_sha256"]
    assert render["gif"]["first_last_identical"] and render["gif"]["duration_ms"]==30020
    assert render["gif"]["width_per_leg"]==[1/16,1]
    summary=dict(grid=16,fresh_targets=8,fresh_tiles=768,ordinary_tiles_per_final_board=255,
                 images=files,edges=render["edges"],gif=render["gif"],quality=quality["stats"],
                 color=render["diagnostics"],peak_reserved_gib=gen["peak_reserved_gib"])
    (ROOT/"verified.json").write_text(json.dumps(summary,indent=2)+"\n")
    bundle=ROOT/"reproduction.zip"
    with ZipFile(bundle,"w",ZIP_DEFLATED,compresslevel=3) as z:
        def include(p):z.write(p,p.relative_to(PROJECT))
        for p in sorted((PROJECT/"sdxl").iterdir()):
            if p.is_file() and p.suffix in (".py",".md",".txt"):include(p)
        for name in ("LICENSE","requirements-sdxl.txt"):include(PROJECT/name)
        for p in sorted(ROOT.iterdir()):
            if p.is_file() and (p.suffix in (".py",".md",".json") or p.name in ("opening.png","opening.safetensors")):include(p)
        for p in sorted((ROOT/"baseline").iterdir()):
            if p.suffix in (".png",".json"):include(p)
        for p in sorted((ROOT/"targets").glob("*.json")):include(p)
        for p in sorted(ROOT.glob("*/stage.json")):include(p)
    with ZipFile(bundle) as z:assert z.testzip() is None
    print("Verified",len(seeds),"native samples and three 16384-square boards.",flush=True)
if __name__=="__main__":main()
