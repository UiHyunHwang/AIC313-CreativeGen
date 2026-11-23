"""Three SDXL grids. The selected native latent is the following grid's target."""
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"dualscale_pilot"))
from run import DisplayJudge, LUMA, predict, steer
from sdxl.hero_cycle import Verifier, Tail, read_json, write_json, thumb
from sdxl.hero_cycle_render import digest
from sdxl.config import RunConfig
from sdxl.run_sdxl_mosaic import _configure_determinism,_load_pipeline,_decode_latent

GRID=8
OPEN="A centered front-facing portrait of Spider-Man, his red webbed mask and two large white eyes filling the frame against a charcoal background."
HEROES=[
 ("spider_man","iron_man",["Iron Man in red and gold armor","Iron Man helmet, glowing eyes","Iron Man in an action pose"],
  ["Iron Man in red and gold armor","Iron Man with a gold faceplate and white glowing eyes"],["an empty background","a tiny distant speck","a red cloth mask with spiderweb markings"]),
 ("iron_man","captain_america",["Captain America holding his shield","Captain America in a blue helmet, waist-up","Captain America running with his shield"],
  ["Captain America in blue uniform with a star on his chest","Captain America in his blue helmet holding his shield"],["a shield or flag alone without a person","an empty background","a tiny distant speck"]),
 ("captain_america","spider_man",["Spider-Man in his red and blue suit","Spider-Man wearing a red webbed mask, waist-up","Spider-Man crouching"],
  ["Spider-Man in a red and blue suit","Spider-Man wearing a red webbed mask with white eyes"],["an insect or spiderweb without a superhero","a tiny distant speck","an empty background"])]
class GridSampler: pass

def encode(pipe,prompt):
    pos,neg,pp,npool=pipe.encode_prompt(prompt=prompt,negative_prompt="",device="cuda",do_classifier_free_guidance=True)
    ids=pipe._get_add_time_ids((1024,1024),(0,0),(1024,1024),dtype=pos.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).cuda()
    return dict(prompt_embeds=pos,negative_prompt_embeds=neg,pooled_prompt_embeds=pp,negative_pooled_prompt_embeds=npool,
        add_time_ids=ids,negative_add_time_ids=ids)

def synth(sampler,initial,seed,condition,target):
    clean=initial.contiguous().cuda()
    eps=torch.randn(clean.shape,dtype=clean.dtype,generator=torch.Generator().manual_seed(seed)).cuda()
    tail=sampler.tails[12]
    z=sampler.pipe.scheduler.add_noise(clean,eps,tail.times[:1])
    for k,t in enumerate(tail.times):
        u=predict(sampler.pipe.unet,z,t,condition,True)
        direction=predict(sampler.pipe.unet,z,t,condition,False)-u
        if k==5:
            z,trace=steer(sampler,z,t,u,direction,clean,thumb(target)@LUMA,2.)
        else:z,_=tail.step(u+5.*direction,t,z)
    return z,_decode_latent(sampler.pipe,z),trace

def parent(latent_path,png_path):
    latent=load_file(str(latent_path))["latents"].cuda().float()
    enlarged=F.interpolate(latent,size=(GRID*128,GRID*128),mode="bicubic",align_corners=False).half().cpu()
    with Image.open(png_path) as im:image=im.convert("RGB")
    return enlarged,image

@torch.inference_mode()
def main():
    _configure_determinism(True)
    recipe=ROOT/"recipe.json"
    cfg=replace(RunConfig.read(recipe),gpu_memory_limit_gib=None,min_free_gpu_gib=0.)
    sampler=GridSampler();sampler.pipe=pipe=_load_pipeline(cfg,True)
    zpath=ROOT/"opening.safetensors";ppath=ROOT/"opening.png"
    if not(zpath.exists() and ppath.exists()):
        z=pipe(prompt=OPEN,negative_prompt="",width=1024,height=1024,num_inference_steps=20,guidance_scale=5.,eta=0.,
               generator=torch.Generator().manual_seed(42),output_type="latent").images
        _decode_latent(pipe,z,"cpu-tiles").save(ppath)
        save_file({"latents":z.cpu().contiguous()},str(zpath))
    conditions={text:encode(pipe,text) for _,_,prompts,_,_ in HEROES for text in prompts}
    pipe.maybe_free_model_hooks();pipe.remove_all_hooks()
    pipe.text_encoder.to("cpu");pipe.text_encoder_2.to("cpu")
    pipe.unet.to("cuda");pipe.vae.to("cuda")
    sampler.tails={12:Tail(pipe.scheduler,12)}
    stages=[]
    for j,(hero,leaf,prompts,pos,neg) in enumerate(HEROES):
        directory=ROOT/hero;directory.mkdir(exist_ok=True)
        clean,target=parent(zpath,ppath)
        zhash=digest(zpath)
        sampler.judge=DisplayJudge(dict(positive=pos,negative=neg))
        # Score close-up suitability separately from basic character recognizability.
        key=["iron","captain","spider"][j]
        records=[]
        for i in range(GRID**2):
            r,c=divmod(i,GRID);prefix=directory/f"r{r+1:02}_c{c+1:02}"
            png,latent,meta=[prefix.with_suffix(s) for s in (".png",".safetensors",".json")]
            seed=8620000+1000*j+i;prompt=prompts[(r+2*c)%3]
            if meta.exists() and png.exists() and latent.exists():
                old=read_json(meta)
                if old["parent_sha256"]==zhash and old["seed"]==seed and old["prompt"]==prompt and old["png_sha256"]==digest(png) and old["latent_sha256"]==digest(latent):
                    records.append(old);continue
            box=(128*c,128*r,128*c+128,128*r+128);ref=target.crop(box)
            z,im,trace=synth(sampler,clean[...,box[1]:box[3],box[0]:box[2]],seed,conditions[prompt],ref)
            scores=sampler.judge.score(leaf,im,ref)
            close=Verifier.score(sampler.judge,key,im,ref)
            im.save(png,compress_level=2)
            save_file({"latents":z.cpu().contiguous()},str(latent))
            record=dict(index=i,row=r+1,column=c+1,seed=seed,prompt=prompt,
                prefix=str(prefix.relative_to(ROOT)),parent_sha256=zhash,png_sha256=digest(png),latent_sha256=digest(latent),
                portrait=close["clip"],trace=trace,**scores)
            write_json(meta,record);records.append(record)
            if (i+1)%4==0:print(hero,leaf,i+1,"/64",flush=True)
        best=max(records,key=lambda v:(v["portrait"]+.5*min(v["native"],v["display44"])-.15*v["color_error"],-v["index"]))
        stage=dict(subject=hero,leaves=leaf,source=str(ppath.relative_to(ROOT)),source_latent=str(zpath.relative_to(ROOT)),
                   source_sha256=zhash,tiles=records,prompts=prompts,positive=pos,negative=neg)
        if j<2:stage["outgoing"]=best["index"]
        stages.append(stage);write_json(directory/"stage.json",stage)
        zpath=(ROOT/best["prefix"]).with_suffix(".safetensors");ppath=zpath.with_suffix(".png")
        print("Stage finished",hero,best["row"],best["column"],flush=True)
    write_json(ROOT/"generation.json",dict(grid=GRID,opening_prompt=OPEN,target_seed=42,stages=stages,
        tile_steps=12,branch_step=6,weight=2.,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))

if __name__=="__main__":main()
