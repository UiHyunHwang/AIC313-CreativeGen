"""Generate SDXL alternatives for low-readability leaves on the existing opening board."""
from pathlib import Path
from dataclasses import replace
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import load_file
from sdxl.config import RunConfig
from sdxl.hero_cycle import Experiment, Tail, Verifier, patch, thumb, read_json, write_json
from sdxl.run_sdxl_mosaic import _configure_determinism, _load_pipeline
from sdxl.hero_cycle_render import digest

PROMPTS = ("Spider-Man in his red webbed suit, waist-up action pose",
           "Spider-Man crouching, red mask and white eyes",
           "Spider-Man swinging in his red and blue suit")
DEFAULT = Path("outputs/inward_cycle")

class Judge(Verifier):
    def __init__(self, review):
        super().__init__()
        positive, negative = review["positive"], review["negative"]
        self.count = len(positive)
        enc = self.proc(text=positive+negative, return_tensors="pt", padding=True)
        self.text = torch.nn.functional.normalize(self.model.get_text_features(**enc))

    def score(self, hero, image, reference):
        views = [image, image.resize((32, 32), Image.Resampling.BOX)]
        feats = self.model.get_image_features(**self.proc(images=views, return_tensors="pt"))
        sims = torch.nn.functional.normalize(feats) @ self.text.T
        margins = sims[:, :self.count].amax(1) - sims[:, self.count:].amax(1)
        return dict(native=float(margins[0]), small=float(margins[1]), margin=float(margins.mean()),
                    error=float(np.abs(thumb(image)-thumb(reference)).mean()))

class Sampler(Experiment):
    def __init__(self, root, archive, judge, recipe):
        self.root, self.hero, self.judge = root, "spider", judge
        self.cfg = replace(RunConfig.read(recipe), gpu_memory_limit_gib=None, min_free_gpu_gib=0.)
        self.pipe = _load_pipeline(self.cfg, True)
        self.conditions = {}
        for prompt in PROMPTS:
            pos, neg, pool, npool = self.pipe.encode_prompt(prompt=prompt, negative_prompt="", device="cuda", do_classifier_free_guidance=True)
            ids = self.pipe._get_add_time_ids((1024,1024),(0,0),(1024,1024),dtype=pos.dtype,
                        text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim).cuda()
            self.conditions[prompt] = dict(prompt_embeds=pos, negative_prompt_embeds=neg, pooled_prompt_embeds=pool,
                        negative_pooled_prompt_embeds=npool, add_time_ids=ids, negative_add_time_ids=ids)
        self.pipe.maybe_free_model_hooks()
        self.pipe.remove_all_hooks()
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        self.pipe.unet.to("cuda")
        self.pipe.vae.to("cuda")
        self.tails = {12:Tail(self.pipe.scheduler,12)}
        latent = load_file(str(archive/"opening.safetensors"))["latents"]
        self.clean = torch.nn.functional.interpolate(latent.cuda().float(),size=(2048,2048),mode="bicubic",align_corners=False).half().cpu()
        with Image.open(archive/"opening_target.png") as im:
            self.ref = im.convert("RGB")

@torch.inference_mode()
def run(root, prior):
    _configure_determinism(True)
    archive = root.parent/"hero_cycle"
    base = root.parent/"hero_cycle_varied"
    raw = base/read_json(base/"report.json")["raw_runs"][0]
    assert digest(archive/"opening_target.png") == digest(raw/"target.png")
    review = read_json(prior/"spider_review.json")
    old_chosen = {r["index"]:r for r in read_json(prior/"quality.json")["selected"]}
    judge = Judge(review)
    recipe = archive/"outer_recipe.json"
    sampler = Sampler(root, archive, judge, recipe if recipe.exists() else raw/"config.json")
    baseline, chosen, attempts = [], [], []
    for item in review["ranked"]:
        name = item["tile"]
        i = int(name[1:4])*16+int(name[6:])
        old = old_chosen.get(i)
        src = (archive/old["file"]).with_suffix(".png") if old else raw/"tiles"/(name+".png")
        with Image.open(src) as im:
            record = dict(index=i, **judge.score("spider",im,patch(sampler.ref,i)))
        baseline.append(record)
    weak = sorted((v for v in baseline if min(v["native"],v["small"])<0),key=lambda r:(r["margin"],r["index"]))
    for k, before in enumerate(weak):
        i = before["index"]
        best = before
        for attempt in range(3):
            prompt = PROMPTS[(i+attempt)%len(PROMPTS)]
            seed = 5100000+i*10+attempt
            cand = sampler.sample(root/"samples",i,prompt,12,seed)
            attempts.append(cand)
            if (cand["margin"]>=before["margin"]+.008 and min(cand["native"],cand["small"])>=min(before["native"],before["small"])+.005
                and cand["error"]<=before["error"]+.03 and cand["margin"]-.24*cand["error"]>best["margin"]-.24*best["error"]):
                best = cand
            if min(best["native"],best["small"])>=.012: break
        if best is not before: chosen.append(best)
        if (k+1)%10==0: print("Spider-Man positions",k+1,"/",len(weak),"accepted",len(chosen),flush=True)
    result = {v["index"]:v for v in chosen}
    def stats(rows):
        return dict(mean_below_zero=sum(v["margin"]<0 for v in rows),
                    either_below_zero=sum(min(v["native"],v["small"])<0 for v in rows))
    write_json(root/"spider.json",dict(eligible=len(weak),attempted=len(attempts),before=stats(baseline),
        after=stats([result.get(v["index"],v) for v in baseline]),selected=chosen,attempts=attempts,baseline=baseline,
        prompts=PROMPTS,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    sheet=Image.new("RGB",(4*190,4*210),"white")
    pen=ImageDraw.Draw(sheet)
    for j,v in enumerate(weak[:8]):
        i=v["index"]; original=old_chosen.get(i)
        src=(archive/original["file"]).with_suffix(".png") if original else raw/"tiles"/f"r{i//16:03}_c{i%16:03}.png"
        dst=(root/result[i]["file"]).with_suffix(".png") if i in result else src
        for after,file in enumerate((src,dst)):
            x=(j%4)*190;y=(2*(j//4)+after)*210
            with Image.open(file) as im:sheet.paste(im.resize((190,190),Image.Resampling.LANCZOS),(x,y+20))
            pen.text((x+4,y+2),f"{'After' if after else 'Before'}  ({i//16+1},{i%16+1})",fill="black")
    sheet.save(root/"spider_comparison.jpg",quality=94)
    print("Finished",len(chosen),len(attempts),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=DEFAULT)
    p.add_argument("--prior",type=Path,default=Path("outputs/varied_detour_quality"))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);run(a.output,a.prior)
