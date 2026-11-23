"""Reproducible three-level PhotoQuilt experiment on a single GPU. Run via -m sdxl.hero_cycle."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file
from transformers import CLIPModel, CLIPProcessor

from .config import RunConfig
from .run_sdxl_mosaic import _configure_determinism, _decode_latent, _environment, _load_pipeline
from .tiling import tiled_tail_denoise

CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
HEROES = {
    "spider": ("spider man", "Spider-Man, close-up portrait, red webbed mask and large white eyes",
              ["a close-up portrait of Spider-Man's red webbed mask and large white eyes",
               "a large, clearly visible Spider-Man facing the viewer, his head and shoulders filling the picture"],
              ["a distant tiny superhero against a wide background", "a spider insect or spiderweb without a person",
               "an abstract blurry patch with no recognizable subject", "a small full-body figure doing a stunt in a landscape"]),
    "captain": ("captain america", "Captain America, close-up portrait, blue helmet with a white A",
                ["Captain America with his round star shield and a blue helmet",
                 "a clear portrait of Marvel superhero Captain America, red white and blue shield"],
                ["an abstract patch of texture", "a tiny figure in an empty landscape",
                 "Spider-Man with a red webbed mask", "a flag without a superhero"]),
    "iron": ("iron man", "Iron Man, close-up portrait of his red and gold helmet with glowing white eyes",
             ["a close-up portrait of Iron Man wearing his red and gold metal helmet", "Iron Man's faceplate with two glowing white eyes filling the frame"],
             ["an abstract patch of texture", "a tiny figure in an empty landscape", "a spiderweb", "an empty room without a superhero"]),
}


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def thumb(im, side=4):
    return np.asarray(im.convert("RGB").resize((side, side), Image.Resampling.BOX), dtype=np.float32) / 255


def patch(im, index):
    row, column = divmod(index, 16)
    side = im.width // 16
    return im.crop((column*side, row*side, (column+1)*side, (row+1)*side))


class Verifier:
    def __init__(self):
        opts = dict(revision=CLIP_REV, local_files_only=True)
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID, use_fast=False, **opts)
        self.model = CLIPModel.from_pretrained(CLIP_ID, **opts).eval()
        self.prompts = {}
        for hero, (_, _, positive, negative) in HEROES.items():
            toks = self.proc(text=positive+negative, return_tensors="pt", padding=True)
            self.prompts[hero] = torch.nn.functional.normalize(self.model.get_text_features(**toks))

    def score(self, hero, image, reference):
        inputs = self.proc(images=[image, image.resize((32,32), Image.Resampling.BOX)], return_tensors="pt")
        features = torch.nn.functional.normalize(self.model.get_image_features(**inputs))
        sims = features @ self.prompts[hero].T
        margins = sims[:,:2].mean(1) - sims[:,2:].amax(1)
        margin = margins.mean().item()
        error = float(np.abs(thumb(image)-thumb(reference)).mean())
        return dict(clip=margin, full_clip=margins[0].item(), small_clip=margins[1].item(),
                    rgb_mae=error, score=margin-0.2*error)


class Tail:
    """Unclipped epsilon DDIM, eta=0, using explicit adjacent times (same start for 12 and 20)."""
    def __init__(self, scheduler, updates, device="cuda"):
        assert scheduler.config.prediction_type == "epsilon" and not scheduler.config.clip_sample
        self.times = torch.linspace(836, 0, updates, device=device).round().long()
        indices = self.times.cpu().tolist()
        self.alphas = {t: (scheduler.alphas_cumprod[t], scheduler.alphas_cumprod[indices[i+1]]
                       if i+1 < len(indices) else scheduler.final_alpha_cumprod) for i,t in enumerate(indices)}

    def scale_model_input(self, x, t):
        return x

    def step(self, eps, t, x, **kwargs):
        alpha, previous = self.alphas[int(t)]
        clean = (x - (1-alpha)**0.5 * eps) / alpha**0.5
        return previous**0.5 * clean + (1-previous)**0.5 * eps, clean


class Experiment:
    def __init__(self, root, recipe):
        self.root = root
        self.cfg = replace(RunConfig.read(recipe), gpu_memory_limit_gib=None, min_free_gpu_gib=0.)
        self.cfg.validate(replay=True)
        self.pipe = _load_pipeline(self.cfg, True)
        self.judge = Verifier()
        self.conditions = {}
        write_json(root/"environment.json", _environment(self.cfg,self.pipe))
        write_json(root/"protocol.json", dict(heroes=HEROES, clip=[CLIP_ID,CLIP_REV],
            score="mean CLIP margin at native and 32px minus 0.2 times 4x4 RGB MAE (RGB in [0,1])",
            pilots="eight luminance quantiles, two prompts x 12/20 updates at t=836; matched full-canvas noise seed 43",
            seeds="weakest 64 by joint score: two additional seeds; third if below initial median",
            candidate_rule="maximize joint score subject to base RGB error + 0.04 and nondecreasing CLIP",
            zoom_rule="highest mean full/thumbnail CLIP score after rerolls; ties use smallest index",
            return_rule="reserve one slot before Iron Man sampling by lowest 4x4 RGB error to opening mosaic preview",
            prompt_authorship="Short Spider-Man and Captain America prompts from user. Global and portrait prompts brainstormed with Codex.",
            references=["https://arxiv.org/abs/2606.30968v2","https://github.com/KooroshRH/PhotoQuilt",
                        "https://arxiv.org/abs/2307.01952","https://arxiv.org/abs/2010.02502",
                        "https://huggingface.co/"+self.cfg.model_id,"https://huggingface.co/"+CLIP_ID,
                        "https://github.com/huggingface/diffusers","https://github.com/huggingface/transformers"]))

    def target(self):
        path = self.root/"opening.safetensors"
        if not path.exists():
            z = self.pipe(prompt=self.cfg.prompt, negative_prompt="", height=1024, width=1024,
                num_inference_steps=20, guidance_scale=5., eta=0.,
                generator=torch.Generator().manual_seed(42), output_type="latent").images
            _decode_latent(self.pipe,z,"cpu-tiles").save(self.root/"opening_target.png")
            save_file({"latents":z.cpu().contiguous()},str(path))
        return load_file(str(path))["latents"]

    def prepare(self):
        for short, portrait, _, _ in HEROES.values():
            for prompt in (short, portrait):
                pos,neg,pool,npool = self.pipe.encode_prompt(prompt=prompt,negative_prompt="",device="cuda",do_classifier_free_guidance=True)
                ids=self.pipe._get_add_time_ids((1024,1024),(0,0),(1024,1024),dtype=pos.dtype,
                     text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim).cuda()
                self.conditions[prompt]=dict(prompt_embeds=pos,negative_prompt_embeds=neg,
                    pooled_prompt_embeds=pool,negative_pooled_prompt_embeds=npool,add_time_ids=ids,negative_add_time_ids=ids)
        self.pipe.maybe_free_model_hooks()
        self.pipe.remove_all_hooks()
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        self.pipe.unet.to("cuda")
        self.pipe.vae.to("cuda")
        self.tails={n:Tail(self.pipe.scheduler,n) for n in (12,20)}

    def sample(self, folder, index, prompt, steps, seed):
        folder.mkdir(parents=True,exist_ok=True)
        stem=f"p{int(prompt==HEROES[self.hero][1])}_n{steps}_i{index:03}_s{seed}"
        path=folder/stem
        if path.with_suffix(".json").exists():
            return read_json(path.with_suffix(".json"))
        row, col = divmod(index,16)
        region=(...,slice(row*128,(row+1)*128),slice(col*128,(col+1)*128))
        clean=self.clean[region].contiguous().cuda()
        if seed==43:
            noise=self.noise[region].contiguous().cuda()
        else:
            noise=torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed),dtype=clean.dtype).cuda()
        tail=self.tails[steps]
        initial=self.pipe.scheduler.add_noise(clean,noise,tail.times[:1])
        final=tiled_tail_denoise(latents=initial,timesteps=tail.times,unet=self.pipe.unet,scheduler=tail,
              guidance_scale=5.,tile_latent=128,tile_batch=1,serial_cfg=True,progress=False,**self.conditions[prompt])
        image=_decode_latent(self.pipe,final)
        record=dict(index=index, prompt=prompt, steps=steps, seed=seed, file=str(path.relative_to(self.root)),
                    **self.judge.score(self.hero,image,patch(self.ref,index)))
        image.save(path.with_suffix(".png"),compress_level=2)
        save_file({"latents":final.cpu().contiguous()},str(path.with_suffix(".safetensors")))
        write_json(path.with_suffix(".json"),record)
        return record

    def controls(self, folder):
        records=[]
        # Independent native portraits from pure noise, paired across both prompts.
        for seed in (42,43):
            for prompt in HEROES[self.hero][:2]:
                self.pipe.scheduler.set_timesteps(20,device="cuda")
                ts=self.pipe.scheduler.timesteps
                noise=torch.randn((1,4,128,128),generator=torch.Generator().manual_seed(seed),dtype=torch.float16).cuda()
                final=tiled_tail_denoise(latents=noise,timesteps=ts,unet=self.pipe.unet,scheduler=self.pipe.scheduler,
                    guidance_scale=5.,tile_latent=128,tile_batch=1,serial_cfg=True,progress=False,**self.conditions[prompt])
                im=_decode_latent(self.pipe,final)
                name=f"control_{seed}_{int(prompt==HEROES[self.hero][1])}.png"
                im.save(folder/name)
                records.append(dict(file=name,seed=seed,prompt=prompt,clip=self.judge.score(self.hero,im,im)["clip"]))
        return records

    def level(self, hero, z, ref, opening=None):
        folder=self.root/hero
        folder.mkdir(exist_ok=True)
        if (folder/"selected.json").exists():
            return read_json(folder/"selected.json")
        self.hero,self.ref=hero,ref
        self.clean=torch.nn.functional.interpolate(z.cuda().float(),size=(2048,2048),mode="bicubic",align_corners=False).half().cpu()
        self.noise=torch.randn(self.clean.shape,generator=torch.Generator().manual_seed(43),dtype=self.clean.dtype)
        candidates=folder/"samples"
        pilots=folder/"pilot.json"
        if hero!="iron":
            if not pilots.exists():
                controls=self.controls(folder)
                by_light=sorted(range(256),key=lambda i: float(np.mean(thumb(patch(ref,i)))))
                indices=[by_light[i] for i in (15,47,79,111,143,175,207,239)]
                groups=[]
                for prompt in HEROES[hero][:2]:
                    for steps in (12,20):
                        samples=[self.sample(candidates,i,prompt,steps,43) for i in indices]
                        values=[v["score"] for v in samples]
                        groups.append(dict(prompt=prompt,steps=steps,mean=float(np.mean(values)),q25=float(np.quantile(values,.25)),
                            clip_mean=float(np.mean([v["clip"] for v in samples])),error_mean=float(np.mean([v["rgb_mae"] for v in samples])),
                            samples=[v["file"] for v in samples]))
                        print(hero,"pilot",prompt,steps,groups[-1]["mean"],flush=True)
                best=max(groups,key=lambda g:(g["mean"]+g["q25"])/2)
                write_json(pilots,dict(indices=indices,groups=groups,best=best,controls=controls))
            winner=read_json(pilots)["best"]
            prompt,steps=winner["prompt"],winner["steps"]
        else:
            prompt,steps=HEROES[hero][1],12
        portal=None
        if opening is not None:
            portal=min(range(256),key=lambda i:float(np.abs(thumb(patch(ref,i))-thumb(opening)).mean()))
        indices=[i for i in range(256) if i!=portal]
        base={}
        for i in indices:
            base[i]=self.sample(candidates,i,prompt,steps,43)
            if (i+1)%32==0: print(hero,"first pass",i+1,"/ 256",flush=True)
        final=base.copy()
        median=float(np.median([r["score"] for r in base.values()]))
        if hero!="iron":
            for k,i in enumerate(sorted(indices,key=lambda i:base[i]["score"])[:64]):
                for j in range(3):
                    if j==2 and final[i]["score"]>=median: break
                    seed=1000000*(1 if hero=="spider" else 2)+i*10+j
                    test=self.sample(candidates,i,prompt,steps,seed)
                    if (test["rgb_mae"]<=base[i]["rgb_mae"]+.04 and test["clip"]>=base[i]["clip"]
                            and test["score"]>final[i]["score"]):
                        final[i]=test
                if (k+1)%16==0: print(hero,"automatic selection",k+1,"/ 64",flush=True)
        best=max(final.values(),key=lambda v:(v["clip"],-v["index"]))
        result=dict(hero=hero,prompt=prompt,steps=steps,reserved=portal,best=best,
                    base=list(base.values()),tiles=list(final.values()))
        write_json(folder/"selected.json",result)
        return result


def preview(root, result):
    canvas=Image.new("RGB",(1024,1024))
    for record in result["tiles"]:
        i=record["index"]
        with Image.open((root/record["file"]).with_suffix(".png")) as im:
            canvas.paste(im.resize((64,64),Image.Resampling.BOX),(i%16*64,i//16*64))
    return canvas


@torch.inference_mode()
def run(root,recipe):
    root.mkdir(exist_ok=True,parents=True)
    _configure_determinism(True)
    started=time.time()
    experiment=Experiment(root,recipe)
    z=experiment.target()
    experiment.prepare()
    with Image.open(root/"opening_target.png") as im:
        ref=im.convert("RGB")
    opening=None
    for hero in HEROES:
        result=experiment.level(hero,z,ref,opening if hero=="iron" else None)
        print(hero,"complete, portal",result["best"]["index"],result["reserved"],flush=True)
        if hero=="spider":
            opening=preview(root,result)
        best=root/result["best"]["file"]
        z=load_file(str(best.with_suffix(".safetensors")))["latents"]
        with Image.open(best.with_suffix(".png")) as im:
            ref=im.convert("RGB")
    write_json(root/"generation.json",dict(status="complete",seconds=time.time()-started,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("outputs/hero_cycle"))
    p.add_argument("--recipe",type=Path,default=Path(__file__).with_name("hero_cycle_recipe.json"))
    a=p.parse_args()
    run(a.output,a.recipe)
