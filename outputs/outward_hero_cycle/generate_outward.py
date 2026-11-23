"""Generate the two boards needed to reveal Spider-Man, Iron Man and Captain America from near to far."""
from pathlib import Path
import sys
from dataclasses import replace
import torch

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"spider_iron_captain"
sys.path.insert(0,str(PREV))
from generate import (HEROES,GRID,GridSampler,DisplayJudge,Tail,RunConfig,parent,encode,synth,
                      digest,read_json,write_json,_configure_determinism,_load_pipeline)
@torch.inference_mode()
def main():
    _configure_determinism(True)
    cfg=replace(RunConfig.read(PREV/"recipe.json"),gpu_memory_limit_gib=None,min_free_gpu_gib=0.)
    sampler=GridSampler();sampler.pipe=pipe=_load_pipeline(cfg,True)
    recipes=HEROES[:2]
    conditions={p:encode(pipe,p) for _,_,prompts,_,_ in recipes for p in prompts}
    pipe.maybe_free_model_hooks();pipe.remove_all_hooks()
    pipe.text_encoder.to("cpu");pipe.text_encoder_2.to("cpu")
    pipe.unet.to("cuda");pipe.vae.to("cuda");sampler.tails={12:Tail(pipe.scheduler,12)}
    previous=read_json(PREV/"generation.json")["stages"]
    stages=[]
    for k,(hero,origin,recipe) in enumerate((("captain_america",previous[2]["source"],recipes[0]),
                                             ("spider_man",previous[0]["source"],recipes[1]))):
        _,leaf,prompts,pos,neg=recipe
        folder=ROOT/hero;folder.mkdir(exist_ok=True)
        png=PREV/origin;latent=png.with_suffix(".safetensors")
        canvas,target=parent(latent,png);latent_sha=digest(latent)
        sampler.judge=DisplayJudge(dict(positive=pos,negative=neg))
        rows=[]
        for i in range(GRID**2):
            r,c=divmod(i,GRID);seed=8720000+1000*k+i;prompt=prompts[(r+2*c)%len(prompts)]
            path=folder/f"r{r+1:02}_c{c+1:02}.png";meta=path.with_suffix(".json")
            if meta.exists() and path.exists():
                row=read_json(meta)
                if row["source_latent"]==latent_sha and row["seed"]==seed and row["prompt"]==prompt and digest(path)==row["sha256"]:
                    rows.append(row);continue
            x,y=128*c,128*r
            ref=target.crop((x,y,x+128,y+128))
            _,im,trace=synth(sampler,canvas[...,y:y+128,x:x+128],seed,conditions[prompt],ref)
            im.save(path,compress_level=2)
            row=dict(index=i,seed=seed,prompt=prompt,file=str(path.relative_to(ROOT)),sha256=digest(path),
                     source_latent=latent_sha,decision=trace,**sampler.judge.score(leaf,im,ref))
            write_json(meta,row);rows.append(row)
            if (i+1)%8==0:print(hero,leaf,i+1,"/ 64",flush=True)
        stage=dict(subject=hero,leaves=leaf,source=origin,source_png=digest(png),source_latent=latent_sha,
                   prompts=prompts,positive=pos,negative=neg,tiles=rows)
        write_json(folder/"stage.json",stage);stages.append(stage)
    write_json(ROOT/"generation.json",dict(grid=GRID,new_stages=stages,steps=12,branch_step=6,
               tone_weight=2.,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
if __name__=="__main__":main()
