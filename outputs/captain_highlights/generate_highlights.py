"""Extend the subject-first schedule with a smooth spatial weight derived from the SDXL target."""
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
FIRST = ROOT.parent / "captain_firstpass"
from outputs.captain_firstpass import experiment as exp
from PIL import Image

def weight(ref):
    # Work on area averages so isolated specular pixels do not control a region.
    luminance = exp.thumb(ref, 16) @ exp.LUMA
    value = torch.tensor(np.clip((luminance-.63)/.16, 0, 1), dtype=torch.float32)[None,None]
    value = F.max_pool2d(value, 3, stride=1, padding=1)
    return F.interpolate(value, (128,128),mode="bilinear",align_corners=False)

def sample(pipe, tail, crop, seed, condition, mask):
    clean = crop.contiguous().cuda()
    mask = mask.cuda()
    noise = torch.randn(clean.shape, dtype=clean.dtype,generator=torch.Generator().manual_seed(seed)).cuda()
    base = torch.lerp(exp.lowpass(clean,2), clean.float(), mask).to(clean.dtype)
    z = pipe.scheduler.add_noise(base,noise,tail.times[:1])
    for k,t in enumerate(tail.times):
        u = exp.predict(pipe.unet,z,t,condition,True)
        direction = exp.predict(pipe.unet,z,t,condition,False) - u
        z_next,x0 = tail.step(u+(6.5 if k<5 else 5.)*direction,t,z)
        if k in exp.LATE:
            delta = clean.float()-x0.float()
            coarse = exp.lowpass(delta,4)
            anchor = torch.lerp(coarse,1.5*exp.lowpass(delta,16),mask)
            b = float(tail.alphas[int(t)][1])
            z_next = (z_next.float() + b**.5*exp.LATE[k]*anchor).to(z.dtype)
        z = z_next
    return exp._decode_latent(pipe,z)

@torch.inference_mode()
def main():
    prev=exp.read_json(FIRST/"protocol.json")
    assert exp.digest(FIRST/"experiment.py")==prev["generator_sha256"]
    version=hashlib.sha256((exp.digest(Path(__file__))+prev["generator_sha256"]).encode()).hexdigest()
    exp.write_json(ROOT/"protocol.json",dict(prev,generator_sha256=version,base_schedule=prev["generator_sha256"],
        mask=dict(pool=16, onset=.63, ramp=.16, dilation=3, native_coarse=16, multiplier=1.5),
        secondary_gate=dict(min_highlight_recovery=.5,max_raw_board_luma64_extra=.002)))
    stage=exp.read_json(exp.OLD/"spider_man/stage.json")
    png=exp.SOURCE/stage["source"]
    assert exp.digest(png)==stage["source_png"] and exp.digest(png.with_suffix(".safetensors"))==stage["source_latent"]
    exp._configure_determinism(True)
    cfg=exp.replace(exp.RunConfig.read(exp.SOURCE/"recipe.json"),gpu_memory_limit_gib=None,min_free_gpu_gib=0.)
    pipe=exp._load_pipeline(cfg,True)
    conditions={p:exp.encode(pipe,p) for p in stage["prompts"]}
    pipe.maybe_free_model_hooks()
    pipe.remove_all_hooks()
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.unet.to("cuda")
    pipe.vae.to("cuda")
    tail=exp.Tail(pipe.scheduler,12)
    judge=exp.DisplayJudge(dict(positive=stage["positive"],negative=stage["negative"]))
    canvas,target=exp.parent(png.with_suffix(".safetensors"),png)
    generated=0
    for tile in stage["tiles"]:
        i=tile["index"]
        out=ROOT/tile["file"]
        out.parent.mkdir(exist_ok=True)
        first=FIRST/tile["file"]
        record=exp.read_json(first.with_suffix(".json"))
        assert record["generator"]==prev["generator_sha256"] and record["sha256"]==exp.digest(first)
        y,x=128*(i//8),128*(i%8)
        ref=target.crop((x,y,x+128,y+128))
        mask=weight(ref)
        meta=out.with_suffix(".json")
        if out.exists() and meta.exists():
            current=exp.read_json(meta)
            if current["generator"]==version and current["sha256"]==exp.digest(out):
                continue
        active=bool(mask.max()>0)
        if active:
            im=sample(pipe,tail,canvas[...,y:y+128,x:x+128],tile["seed"],conditions[tile["prompt"]],mask)
            # Atomic replace also avoids mutating a hard-linked zero-weight result on a rerun.
            tmp=out.with_suffix(".tmp.png")
            im.save(tmp,compress_level=2)
            os.replace(tmp,out)
            record.update(candidate=exp.metrics(judge,im,ref),sha256=exp.digest(out))
            generated+=1
        elif not out.exists():
            os.link(first,out)
        else:
            assert exp.digest(out)==record["sha256"]
        exp.write_json(meta,dict(record,generator=version,active=active))
        if (i+1)%8==0:print(f"{i+1}/64, {generated} additional diffusions",flush=True)
    print("peak reserved GiB",torch.cuda.max_memory_reserved()/2**30,flush=True)

if __name__=="__main__":
    main()
