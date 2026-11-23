"""Automatically sample weak Captain America leaves before resolving the recursive grid."""
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors import safe_open
from sdxl.config import RunConfig
from sdxl.hero_cycle import Experiment, HEROES, Verifier, Tail, patch, thumb, read_json, write_json
from sdxl.run_sdxl_mosaic import _configure_determinism, _load_pipeline, _environment

PROMPTS = ("Captain America holding his shield", "Captain America in action, blue suit and star shield",
           "Captain America in uniform, cinematic still")
POSITIVE = ("Captain America in his blue uniform carrying a red white and blue shield",
            "Captain America in a clear full-body action pose",
            "a clear face portrait of superhero Captain America wearing his helmet")
NEGATIVE = ("a shield, star or flag alone without a person", "a tiny distant figure in a large empty scene",
            "a blurry abstract patch without a recognizable subject", "a different superhero wearing a spider web suit")


class Judge(Verifier):
    def __init__(self):
        super().__init__()
        text=self.proc(text=list(POSITIVE+NEGATIVE),return_tensors="pt",padding=True)
        self.variety=torch.nn.functional.normalize(self.model.get_text_features(**text))

    def score(self, hero, image, reference):
        inputs=self.proc(images=[image,image.resize((32,32),Image.Resampling.BOX)],return_tensors="pt")
        features=torch.nn.functional.normalize(self.model.get_image_features(**inputs))
        original=features@self.prompts[hero].T
        clip=float((original[:,:2].mean(1)-original[:,2:].amax(1)).mean())
        sims=features@self.variety.T
        negatives=sims[:,len(POSITIVE):]
        margin=float((sims[:,:len(POSITIVE)].amax(1)-negatives.amax(1)).mean())
        error=float(np.abs(thumb(image)-thumb(reference)).mean())
        return dict(clip=clip, variety=margin,rgb_mae=error,score=(clip+margin)/2-.2*error,
                    contrast=NEGATIVE[int(negatives.mean(0).argmax())])


class Sampler(Experiment):
    def __init__(self,root,middle,judge,prompts=PROMPTS):
        self.root=root
        self.cfg=replace(RunConfig.read(middle/"config.json"),min_free_gpu_gib=0.,gpu_memory_limit_gib=None)
        self.pipe=_load_pipeline(self.cfg,True)
        self.judge=judge
        self.hero="captain"
        self.conditions={}
        for prompt in prompts:
            pos,neg,pool,npool=self.pipe.encode_prompt(prompt=prompt,negative_prompt="",device="cuda",do_classifier_free_guidance=True)
            ids=self.pipe._get_add_time_ids((1024,1024),(0,0),(1024,1024),dtype=pos.dtype,
                         text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim).cuda()
            self.conditions[prompt]=dict(prompt_embeds=pos,negative_prompt_embeds=neg,pooled_prompt_embeds=pool,
                                         negative_pooled_prompt_embeds=npool,add_time_ids=ids,negative_add_time_ids=ids)
        self.pipe.maybe_free_model_hooks()
        self.pipe.remove_all_hooks()
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        self.pipe.unet.to("cuda")
        self.pipe.vae.to("cuda")
        self.tails={12:Tail(self.pipe.scheduler,12)}
        source=self.cfg.parent_tile_source
        with safe_open(str(Path(source["run"])/"final_latents.safetensors"),framework="pt") as f:
            r,c=source["row"],source["column"]
            latent=f.get_slice("latents")[:,:,r*128:(r+1)*128,c*128:(c+1)*128]
        self.clean=torch.nn.functional.interpolate(latent.cuda().float(),size=(2048,2048),mode="bicubic",align_corners=False).half().cpu()
        with Image.open(middle/"target.png") as im:
            self.ref=im.convert("RGB")
        write_json(root/"refinement_environment.json",_environment(self.cfg,self.pipe))


@torch.inference_mode()
def refine(root,middle,records,locked):
    _configure_determinism(True)
    judge=Judge()
    with Image.open(middle/"target.png") as f:
        ref=f.convert("RGB")
    base=[]
    for rec in records["tiles"]:
        with Image.open((root/rec["file"]).with_suffix(".png")) as im:
            base.append(dict(**rec,seed=43,prompt="captain america",**judge.score("captain",im,patch(ref,rec["index"]))))
    weak=sorted((r for r in base if r["index"]!=locked and min(r["clip"],r["variety"])<.005),
                key=lambda r:(r["clip"]+r["variety"],r["index"]))
    write_json(root/"baseline.json",dict(weak_indices=[r["index"] for r in weak],tiles=base))
    print("Reviewed",len(base),"tiles;",len(weak),"eligible for SDXL resampling",flush=True)
    sampler=Sampler(root,middle,judge)
    selected=base.copy()
    for k,original in enumerate(weak):
        index=original["index"]
        best=original
        for attempt,prompt in enumerate(PROMPTS):
            seed=4000000+10*index+attempt
            candidate=sampler.sample(root/"candidates",index,prompt,12,seed)
            if (candidate["clip"]>=original["clip"] and candidate["variety"]>=original["variety"]
                and candidate["rgb_mae"]<=original["rgb_mae"]+.035 and candidate["score"]>best["score"]+.002):
                best=candidate
            if min(best["clip"],best["variety"])>=.025: break
        selected[index]=best
        if (k+1)%16==0: print("Reviewed candidates for",k+1,"/",len(weak),flush=True)
    def stats(tiles):
        visible=[r for r in tiles if r["index"]!=locked]
        return dict(count=len(visible),old_clip_below_zero=sum(r["clip"]<0 for r in visible),
                    **{key:dict(mean=float(np.mean([r[key] for r in visible])),p10=float(np.quantile([r[key] for r in visible],.1)))
                       for key in ("clip","variety","rgb_mae")})
    n=sum(r["seed"]!=43 for r in selected)
    summary=dict(prompts=PROMPTS,positive=POSITIVE,negative=NEGATIVE,locked_index=locked,
                 weak_count=len(weak),replaced=n,candidate_count=len(list((root/"candidates").glob("*.json"))),
                 before=stats(base),after=stats(selected),selected=selected,
                 rule="Threshold 0.005 on either verifier margin; same parent/12 updates/t=836. Both margins must match or exceed original; RGB MAE tolerance 0.035. Early stop at both margins >=0.025. Deterministic seeds recorded.")
    write_json(root/"refinement.json",summary)
    print("Selected replacements:",n,summary["before"],summary["after"],flush=True)
    # Show the original eight weakest eligible positions, including any positions kept.
    sheet=Image.new("RGB",(8*160,350),"white")
    draw=ImageDraw.Draw(sheet)
    for j,r in enumerate(weak[:8]):
        for y,v in ((0,r),(175,selected[r["index"]])):
            with Image.open((root/v["file"]).with_suffix(".png")) as im:
                sheet.paste(im.resize((160,148),Image.Resampling.LANCZOS),(160*j,y+22))
            row,col=divmod(r["index"],16)
            draw.text((j*160+3,y+3),f"{'Before' if y==0 else 'After'}  ({row+1},{col+1})",fill="black")
    sheet.save(root/"before_after.jpg",quality=93)
    return dict(tiles=selected),summary
