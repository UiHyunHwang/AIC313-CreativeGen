"""From the repository root, regenerate three SDXL boards from recorded prompts and seeds."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
from sdxl.hero_cycle_render import NODES, cell, digest, movie, read, resolve, write_boards
from third_recipe import child_recipe

HERE = Path(__file__).resolve().parent
REPO = Path.cwd()
ROWS = 16


def coordinate(tile):
    row, col = (int(v[1:]) for v in tile.split("_"))
    return row*ROWS+col


def tile_records(root, run):
    tiles = sorted((run/"tiles").glob("*.png"))
    assert len(tiles) == 256
    outputs = read(run/"manifest.json")["outputs"]
    records = []
    for index, tile in enumerate(tiles):
        assert coordinate(tile.stem) == index
        assert digest(tile) == outputs["tiles/"+tile.name]["sha256"]
        records.append(dict(index=index,file=os.path.relpath(tile.with_suffix(""),root)))
    return dict(tiles=records)


def generate(root, name, recipe):
    config = root/(name+"_replay.json")
    config.write_text(json.dumps(recipe,indent=2))
    dest=root/("generated_"+name)
    done=[m.parent for m in dest.glob("*/manifest.json") if read(m)["status"]=="complete"]
    if not done:
        subprocess.run([sys.executable,"-m","sdxl","--local-files-only","--config",str(config),
                        "--output-dir",str(dest)],check=True)
        done=[m.parent for m in dest.glob("*/manifest.json") if read(m)["status"]=="complete"]
    if len(done)!=1:
        raise RuntimeError(f"Expected one completed {name} run in {dest}")
    return done[0]


def render(root, runs, selections):
    links=[coordinate(s["ranked"][0]["tile"]) for s in selections]
    recipes=[read(run/"config.json") for run in runs]
    for j,(run,cfg,prompt) in enumerate(zip(runs,recipes,("spider man","captain america","iron man"))):
        assert read(run/"manifest.json")["status"]=="complete"
        assert cfg["tile_prompt"]==prompt and cfg["negative_prompt"]=="" and cfg["width"]==cfg["height"]==16384
        if j:
            r,c=divmod(links[j-1],16)
            source=cfg["parent_tile_source"]
            assert (source["row"],source["column"])==(r,c)
            assert source["sha256"]==digest(runs[j-1]/"final_latents.safetensors")
    assert recipes[0]["prompt"]==read(HERE/"opening_recipe.json")["prompt"]
    def signature(im):
        return np.asarray(im.convert("RGB").resize((4,4),Image.Resampling.BOX),dtype=np.float32)
    with Image.open(runs[0]/"mosaic.png") as outer, Image.open(runs[2]/"target.png") as target:
        opening=signature(outer)
        links.append(min(range(256),key=lambda i:float(np.abs(signature(cell(target,i))-opening).mean())))
    print("Verified inputs. Route",links,flush=True)
    for name in NODES:
        (root/name).mkdir(exist_ok=True)
    results=[tile_records(root,run) for run in runs]
    thumbs,iters=resolve(root,results,links)
    pyramids,checks=write_boards(root,results,links,thumbs)
    video=movie(root,pyramids,links,hold_ms=5500)
    report=dict(boards=checks,gif=video,hold_ms=5500,zoom_ms=4500,thumbnail_iterations=iters,
                prompts=[recipes[0]["prompt"]]+[cfg["tile_prompt"] for cfg in recipes],
                raw_runs=[os.path.relpath(run,root) for run in runs],
                sample_counts=dict(opening_target=1,tile_samples_per_level=[256,256,256],directly_rendered_leaves_per_level=[255,255,255]),
                selection="Non-generative CLIP full and 32px close-up margin, best of 256 for each of the two zoom destinations. Return slot chosen by minimum 4x4 RGB distance to opening overview. Seeds retained in source recipes.",
                credits="PhotoQuilt; official SDXL base 1.0; DDIM; Diffusers; openai/clip-vit-base-patch32; PyTorch; Pillow. Opening prompt and scoring text brainstormed with Codex. Code assistance from Codex.")
    (root/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    with ZipFile(root/"reproduction.zip","w",compression=ZIP_DEFLATED) as archive:
        for source in [REPO/"LICENSE",*sorted((REPO/"sdxl").glob("*.py")),REPO/"sdxl/hero_cycle_requirements.txt"]:
            archive.write(source,str(source.relative_to(REPO)))
        for name in ["replay.py","third_recipe.py","rank_zoom.py","opening_recipe.json","spider_selection.json","captain_selection.json","README.md"]:
            archive.write(HERE/name,"experiment/"+name)
        archive.write(root/"report.json","recorded/report.json")
        for j,run in enumerate(runs):
            for fname in ("config.json","environment.json","manifest.json"):
                archive.write(run/fname,f"recorded/{NODES[j]}/{fname}")
    print("Verified",video,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--runs",type=Path,nargs=3)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    choices=[read(HERE/(name+"_selection.json")) for name in ("spider","captain")]
    if args.runs:
        runs=args.runs
    else:
        initial=read(HERE/"opening_recipe.json")
        runs=[generate(args.output,"spider",initial)]
        for name,prompt,choice in zip(("captain","iron"),("captain america","iron man"),choices):
            cfg=child_recipe(runs[-1],choice["ranked"][0]["tile"],prompt)
            runs.append(generate(args.output,name,cfg))
    render(args.output,runs,choices)
