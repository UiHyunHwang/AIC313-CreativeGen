"""Render all 64 native leaves directly from the pipeline's Iron Man target."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file
from PIL import Image

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"dualscale_pilot"))
from run import (PROMPT, WEIGHTS, PROPOSALS, BRANCH_STEPS, LUMA, DisplayJudge, digest,
                 read_json, write_json, refine_spider, sample, thumb, _configure_determinism)

N=8
@torch.inference_mode()
def main():
    _configure_determinism(True)
    archive=ROOT.parent/"hero_cycle"
    review=read_json(ROOT.parent/"varied_detour_quality/spider_review.json")
    judge=DisplayJudge(review)
    refine_spider.PROMPTS=(PROMPT,)
    sampler=refine_spider.Sampler(ROOT,archive,judge,archive/"outer_recipe.json")
    source=archive/"opening.safetensors"
    latent=load_file(str(source))["latents"]
    sampler.clean=F.interpolate(latent.cuda().float(),size=(N*128,N*128),mode="bicubic",align_corners=False).half().cpu()
    folder=ROOT/"tiles"; folder.mkdir(exist_ok=True)
    protocol=dict(grid=N,prompt=PROMPT,sdxl=sampler.cfg.revision,clip=review["revision"],
                  parent_sha256=digest(source),target_sha256=digest(archive/"opening_target.png"),
                  weight=WEIGHTS["tone"],proposals=PROPOSALS,steps=12,branch_steps=sorted(k+1 for k in BRANCH_STEPS),
                  start_timestep=836,seed_offset=8110000)
    protocol_path=ROOT/"protocol.json"
    if protocol_path.exists() and read_json(protocol_path)!=readback(protocol):
        raise RuntimeError("Output folder contains a different generation protocol")
    write_json(protocol_path,protocol)
    rows=[]
    for i in range(N*N):
        r,c=divmod(i,N)
        path=folder/f"r{r+1:02}_c{c+1:02}.png"
        rec_path=path.with_suffix(".json")
        if rec_path.exists() and path.exists():
            record=read_json(rec_path)
            if record["sha256"]==digest(path) and record["seed"]==8110000+i:
                rows.append(record)
                continue
        side=sampler.ref.width//N
        ref=sampler.ref.crop((c*side,r*side,(c+1)*side,(r+1)*side))
        seed=8110000+i
        image,trace=sample(sampler,i,seed,ref,WEIGHTS["tone"],grid_size=N)
        image.save(path,compress_level=2)
        record=dict(index=i,row=r+1,column=c+1,seed=seed,file=str(path.relative_to(ROOT)),
                    sha256=digest(path),trace=trace,luma_mae=float(np.abs((thumb(image)-thumb(ref))@LUMA).mean()),
                    **judge.score("spider",image,ref))
        write_json(rec_path,record);rows.append(record)
        print(f"Generated {i+1}/64",flush=True)
    write_json(ROOT/"generation.json",dict(protocol=protocol,tiles=rows,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))

def readback(obj):
    import json
    return json.loads(json.dumps(obj))

if __name__=="__main__":main()
