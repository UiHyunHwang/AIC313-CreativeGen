"""Scene-aware SDXL generation on a new Iron Man target and low-detail background cells."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from PIL import Image,ImageDraw
from outputs.outward_palette16 import common as core
from outputs.outward_palette16 import generate_boards as first
from outputs.outward_palette16.targets import config,open_rgb,signature,OUT

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/"outward_palette16"
BOX=Image.Resampling.BOX
NEGATIVE="plain white background, white studio backdrop, isolated cutout, faded sticker, empty seamless backdrop, floating figure, product shot"
POSES=[
    ("Captain America bracing his shield","Captain America in his blue helmet, waist-up","Captain America running with his shield"),
    ("Spider-Man in his red and blue suit, leaning against a railing","Spider-Man wearing his red webbed mask, waist-up","Spider-Man crouching on a ledge"),
    ("Iron Man stepping forward in red and gold armor","Iron Man helmet and shoulders, glowing eyes","Iron Man landing on one knee")]
SETTINGS=dict(zip(first.LIGHTS,(
    "in a graphite steel workshop, layered machinery in shadow behind him",
    "in a narrow red-brick passage, warm light bouncing from the walls",
    "inside an amber-lit engine hall, copper pipes and shadowed beams",
    "on a midnight-blue steel catwalk with girders receding behind him",
    "in a silver railway depot, windows casting strong diagonal shadows")))
def properties(ref):
    a=signature(ref,16);light=a@[.2126,.7152,.0722]
    return float(light.mean()),float(light.std()),float(np.ptp(a.mean((0,1))))
def plain(ref):
    m,s,ch=properties(ref)
    return (m>.42 or ch<.11) and s<.075
def tile_ref(image,i):
    r,c=divmod(i,16);return image.crop((c*64,r*64,(c+1)*64,(r+1)*64))
def make_prompt(j,i,ref):
    r,c=divmod(i,16)
    return POSES[j][(r+2*c)%3]+" "+SETTINGS[first.lighting(ref)]+", cinematic still, directional light and deep shadows"
def condition(pipe,p):
    pos,neg,a,b=pipe.encode_prompt(prompt=p,negative_prompt=NEGATIVE,device="cuda",do_classifier_free_guidance=True)
    ids=pipe._get_add_time_ids((1024,1024),(0,0),(1024,1024),dtype=pos.dtype,
                               text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).cuda()
    return dict(encoder_hidden_states=torch.cat([neg,pos]),added_cond_kwargs=dict(text_embeds=torch.cat([b,a]),time_ids=torch.cat([ids,ids])))
def generate(pipe,tail,clean,ref,cond,seed):
    _,contrast,_=properties(ref)
    # In flat parent regions the parent supplies a single color, leaving pose, set,
    # shadow and depth to SDXL. Boundaries keep the contour-preserving schedule.
    flat=plain(ref)
    contour=min(1.,max(0.,(contrast-.035)/.12))
    mask=core.weight(ref).cuda()*contour
    coarse=core.lowpass(clean,1 if flat else 2)
    base=torch.lerp(coarse,clean.float(),mask)*(0.65 if flat else .88)
    eps=torch.randn(clean.shape,dtype=clean.dtype,generator=torch.Generator().manual_seed(seed)).cuda()
    z=pipe.scheduler.add_noise(base.to(clean.dtype),eps,tail.times[:1])
    for k,t in enumerate(tail.times):
        u,v=pipe.unet(torch.cat([z,z]),t,**cond,return_dict=False)[0].chunk(2)
        z_next,x0=tail.step(u+(6.5 if k<5 else 5.)*(v-u),t,z)
        if k in core.LATE:
            delta=clean.float()-x0.float()
            shift=torch.lerp(core.lowpass(delta,1 if flat else 4),1.5*core.lowpass(delta,16),mask)
            gain=float(tail.alphas[int(t)][1])**.5*core.LATE[k]
            z_next=(z_next.float()+gain*shift).to(z.dtype)
        z=z_next
    return core._decode_latent(pipe,z)
def positions(image,j):
    eligible=[i for i in range(256) if j==1 or plain(tile_ref(image,i))]
    eligible.sort(key=lambda i:properties(tile_ref(image,i))[0])
    count=8 if j==1 else 4
    return sorted(eligible[round(q*(len(eligible)-1))] for q in np.linspace(.04,.96,count))
def get_targets():
    prev=core.read_json(BASE/"targets.json")
    files=[OUT/p for p in prev["selected"]]
    files[1]=BASE/"targets/iron_man_3.png"
    sel=dict(prev,selected=[str(p.relative_to(OUT)) for p in files],hashes=[core.digest(p) for p in files],
             latent_hashes=[core.digest(p.with_suffix(".safetensors")) for p in files],final_indices=dict(iron=3,captain=2),
             change="The user preferred integrated backgrounds. The steel-and-crimson industrial helmet replaces the pale studio portrait.")
    core.write_json(ROOT/"targets.json",sel)
    return sel,files
@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--pilot",action="store_true");args=parser.parse_args()
    if not args.pilot:
        assert core.read_json(ROOT/"pilot_review.json")["adopt"],"Pilot review required"
    core._configure_determinism(True)
    sel,files=get_targets();images=[open_rgb(p) for p in files]
    ids=[positions(im,j) for j,im in enumerate(images)]
    canonical=core.digest(Path(__file__))
    cfg=config();pipe=core._load_pipeline(cfg,True);pipe.maybe_free_model_hooks();pipe.remove_all_hooks()
    pipe.text_encoder.to("cuda");pipe.text_encoder_2.to("cuda")
    prompts={make_prompt(j,i,tile_ref(im,i)) for j,im in enumerate(images) for i in (ids[j] if args.pilot else range(256))
             if j==1 or plain(tile_ref(im,i))}
    conditions={p:condition(pipe,p) for p in sorted(prompts)}
    control={}
    if args.pilot:
        for i in ids[1]:
            ref=tile_ref(images[1],i);r,c=divmod(i,16)
            p=first.PROMPTS[1][(r+2*c)%3]+", "+first.lighting(ref)
            control[p]=first.batch_condition(pipe,p)
    pipe.text_encoder.to("cpu");pipe.text_encoder_2.to("cpu");pipe.unet.to("cuda");pipe.vae.to("cuda")
    tail=core.Tail(pipe.scheduler,12)
    stages=[];pairs=[]
    for j,(hero,image,f) in enumerate(zip(sel["heroes"],images,files)):
        folder=ROOT/hero;folder.mkdir(exist_ok=True)
        assert core.digest(f)==sel["hashes"][j]
        canvas=F.interpolate(load_file(str(f.with_suffix(".safetensors")))["latents"].cuda().float(),
                             size=(2048,2048),mode="bicubic",align_corners=False).half()
        records=[]
        for i in (ids[j] if args.pilot else range(256)):
            ref=tile_ref(image,i);r,c=divmod(i,16);seed=9130000+j*1000+i
            out=folder/f"r{r+1:02}_c{c+1:02}.png";meta=out.with_suffix(".json")
            old=BASE/hero/out.name
            active=j==1 or plain(ref)
            crop=canvas[...,128*r:128*(r+1),128*c:128*(c+1)].contiguous()
            prev=core.read_json(meta) if meta.exists() else {}
            if prev.get("generator")==canonical and prev.get("parent")==sel["latent_hashes"][j] and out.exists() and prev["sha256"]==core.digest(out):
                row=prev
            else:
                if active:
                    prompt=make_prompt(j,i,ref)
                    im=generate(pipe,tail,crop,ref,conditions[prompt],seed)
                    tmp=out.with_suffix(".tmp.png");im.save(tmp,compress_level=2);tmp.replace(out)
                else:
                    original=core.read_json(old.with_suffix(".json"))
                    assert original["sha256"]==core.digest(old) and original["seed"]==seed and original["parent"]==sel["latent_hashes"][j]
                    prompt=original["prompt"]
                    if not out.exists():os.link(old,out)
                    assert core.digest(out)==original["sha256"]
                row=dict(index=i,seed=seed,prompt=prompt,file=str(out.relative_to(ROOT)),sha256=core.digest(out),active=active,
                         parent=sel["latent_hashes"][j],generator=canonical)
                core.write_json(meta,row)
            records.append(row)
            if args.pilot:
                compare=old
                if j==1:
                    controls=ROOT/"pilot_controls";controls.mkdir(exist_ok=True)
                    compare=controls/out.name
                    if not compare.exists():
                        p=first.PROMPTS[j][(r+2*c)%3]+", "+first.lighting(ref)
                        first.sample(pipe,tail,crop,ref,control[p],seed).save(compare,compress_level=2)
                pairs.append(dict(hero=hero,index=i,seed=seed,control=str(compare.relative_to(OUT)),scene=str(out.relative_to(OUT))))
            if not args.pilot and (i+1)%32==0:print(hero,i+1,flush=True)
        if not args.pilot:
            stage=dict(subject=hero,leaf=sel["heroes"][(j-1)%3],target=str(f.relative_to(OUT)),tiles=records)
            core.write_json(folder/"stage.json",stage);stages.append(stage)
    if args.pilot:core.write_json(ROOT/"pilot.json",pairs)
    else:core.write_json(ROOT/"generation.json",dict(grid=16,steps=12,native_tile=1024,stages=stages,model=cfg.model_id,
                 revision=cfg.revision,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    print("done",flush=True)
if __name__=="__main__":main()
