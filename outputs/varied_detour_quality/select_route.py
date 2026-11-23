import json
from pathlib import Path
import numpy as np
from PIL import Image
from sdxl.hero_cycle_render import BOX,cell,read
from rank_zoom import rank_tiles,model_id,revision
from third_recipe import child_recipe

def run(root,base):
    root.mkdir(exist_ok=True,parents=True)
    rep=read(base/"report.json")
    src=[(base/p).resolve() for p in rep["raw_runs"]]
    cap=src[2]
    old=rep["boards"][2]
    old=(old["row_one_based"]-1)*16+old["column_one_based"]-1
    def rgb(img,side):return np.asarray(img.convert("RGB").resize((side,side),BOX),dtype=float)/255
    with Image.open(base/"spider/overview.png") as im:
        opening=rgb(im,64);signature=rgb(im,4)
    with Image.open(cap/"mosaic.png") as raw:
        board=rgb(raw.resize((1024,1024),BOX),1024)
    with Image.open(cap/"target.png") as im:
        target=im.convert("RGB")
    costs=[]
    for i in range(256):
        r,c=divmod(i,16);y,x=r*64,c*64
        edges=[]
        if r: edges.append(np.abs(opening[0]-board[y-1,x:x+64]).mean())
        if c: edges.append(np.abs(opening[:,0]-board[y:y+64,x-1]).mean())
        if r<15: edges.append(np.abs(opening[-1]-board[y+64,x:x+64]).mean())
        if c<15: edges.append(np.abs(opening[:,-1]-board[y:y+64,x+64]).mean())
        seam=float(np.mean(edges))
        macro=float(np.abs(signature-rgb(cell(target,i),4)).mean())
        distance=float(np.hypot(r-5,c-5))
        costs.append(dict(index=i,row=r+1,col=c+1,seam=seam,macro=macro,dist=distance,cost=.65*seam+.35*macro))
    # Prefer the suggested area among perceptually similar numerical costs.
    eligible=[v for v in costs if v["index"]!=old]
    best=min(v["cost"] for v in eligible)
    shortlist=[v for v in eligible if v["cost"]<=best+.015]
    choice=min(shortlist,key=lambda v:(v["dist"],v["cost"]))
    previous_choice=choice
    choice=costs[5*16+6]  # User-selected destination, 1-based (6,7).
    positive=["a clearly recognizable close-up of Iron Man wearing his red and gold helmet with glowing white eyes",
              "a detailed portrait of Iron Man, large red and gold faceplate"]
    negative=["an empty abstract texture without a character","a tiny distant figure in a landscape","an unmasked ordinary person"]
    ranked=rank_tiles(sorted((cap/"tiles").glob("*.png")),positive,negative)
    def idx(tile):
        r,c=[int(x[1:]) for x in tile.split("_")];return r*16+c
    hero=next(v for v in ranked if idx(v["tile"]) not in (old,choice["index"]))
    report=dict(previous_choice=previous_choice,old=costs[old],suggested=costs[85],best_cost=best,chosen=choice,all_costs=costs,
                clip_id=model_id,clip_revision=revision,positive=positive,negative=negative,ranked=ranked,hero=hero,
                hero_index=idx(hero["tile"]),inputs=[str(d) for d in src])
    (root/"selection.json").write_text(json.dumps(report,indent=2))
    (root/"recipe.json").write_text(json.dumps(child_recipe(cap,hero["tile"],"iron man"),indent=2))
    print("Return",choice,"old",costs[old],"(6,6)",costs[85],"Hero",hero,flush=True)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--base",type=Path,default=Path("outputs/hero_cycle_varied"));p.add_argument("--output",type=Path,default=Path("outputs/varied_detour"))
    a=p.parse_args();run(a.output,a.base)
