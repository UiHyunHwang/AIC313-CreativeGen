"""Generate only the two sets of leaves affected by inserting Thor into the cycle."""
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from outputs.outward_palette16 import common as core
from outputs.outward_palette16 import generate_boards as first
from outputs.outward_scene16 import scene
from outputs.outward_thor16.portraits import ROOT, OUT, PRIOR, HEROES, config, open_rgb

THOR_POSES=("Marvel's Thor, long blond hair and a red cape, wielding Mjolnir",
            "Marvel's Thor in dark Asgardian armor, blond beard and red cape, waist-up",
            "Marvel's Thor raising his hammer, flowing red cape and long blond hair")
THOR_SETS=dict(zip(first.LIGHTS,(
    "in a shadowed Asgardian stone hall with iron pillars",
    "between crimson banners on a fortress stair",
    "in a firelit forge with copper pillars",
    "on a storm-blue bridge among stone arches",
    "in a silver stone arcade lit through tall windows")))
def tile_spec(hero,i,ref):
    r,c=divmod(i,16)
    pose=(r+2*c)%3
    light=first.lighting(ref)
    flat=scene.plain(ref)
    if hero=="thor":
        prompt=scene.make_prompt(0,i,ref) if flat else first.PROMPTS[0][pose]+", "+light
    elif flat:
        prompt=THOR_POSES[pose]+" "+THOR_SETS[light]+", cinematic still, directional light and deep shadows"
    else:
        prompt=THOR_POSES[pose]+", "+light
    return flat,prompt
@torch.inference_mode()
def main():
    core._configure_determinism(True)
    selected=core.read_json(ROOT/"targets.json")
    cfg=config()
    pipe=core._load_pipeline(cfg,True)
    pipe.maybe_free_model_hooks();pipe.remove_all_hooks()
    images={}
    prompts=set()
    for j in (0,3):
        target=OUT/selected["selected"][j]
        assert core.digest(target)==selected["hashes"][j]
        im=open_rgb(target)
        hero=HEROES[j];images[hero]=im
        prompts.update(tile_spec(hero,i,scene.tile_ref(im,i)) for i in range(256))
    pipe.text_encoder.to("cuda");pipe.text_encoder_2.to("cuda")
    cond={(flat,p):(scene.condition if flat else first.batch_condition)(pipe,p) for flat,p in sorted(prompts)}
    pipe.text_encoder.to("cpu");pipe.text_encoder_2.to("cpu")
    pipe.unet.to("cuda");pipe.vae.to("cuda")
    tail=core.Tail(pipe.scheduler,12)
    version=core.digest(Path(__file__))
    stages=[]
    for j,hero in enumerate(HEROES):
        if j in (1,2):
            old=core.read_json(PRIOR/hero/"stage.json")
            assert old["leaf"]==HEROES[j-1] and len(old["tiles"])==256
            stages.append(dict(old,directory=PRIOR.name))
            continue
        directory=ROOT/hero;directory.mkdir(exist_ok=True)
        filename=selected["selected"][j];latent=(OUT/filename).with_suffix(".safetensors")
        parent_sha=core.digest(latent);assert parent_sha==selected["latent_hashes"][j]
        z=load_file(str(latent))["latents"].cuda().float()
        canvas=F.interpolate(z,size=(2048,2048),mode="bicubic",align_corners=False).half()
        rows=[]
        for i in range(256):
            r,c=divmod(i,16);ref=scene.tile_ref(images[hero],i)
            flat,p=tile_spec(hero,i,ref);seed=9254000+(1000 if j==3 else 0)+i
            dest=directory/f"r{r+1:02}_c{c+1:02}.png";meta=dest.with_suffix(".json")
            row=core.read_json(meta) if meta.exists() else {}
            if not (dest.exists() and row.get("generator")==version and row.get("parent")==parent_sha
                    and row.get("seed")==seed and row.get("prompt")==p and row.get("sha256")==core.digest(dest)):
                crop=canvas[...,r*128:(r+1)*128,c*128:(c+1)*128].contiguous()
                sampler=scene.generate if flat else first.sample
                im=sampler(pipe,tail,crop,ref,cond[flat,p],seed)
                tmp=dest.with_suffix(".tmp.png");im.save(tmp,compress_level=2);tmp.replace(dest)
                row=dict(index=i,file=str(dest.relative_to(ROOT)),seed=seed,prompt=p,scene=flat,
                         generator=version,parent=parent_sha,sha256=core.digest(dest))
                core.write_json(meta,row)
            rows.append(row)
            if (i+1)%32==0:print(hero,i+1,"/256",flush=True)
        stage=dict(subject=hero,leaf=HEROES[j-1],target=filename,directory=ROOT.name,tiles=rows)
        core.write_json(directory/"stage.json",stage);stages.append(stage)
    core.write_json(ROOT/"generation.json",dict(grid=16,stages=stages,steps=12,native_tile=1024,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,model=cfg.model_id,revision=cfg.revision))
if __name__=="__main__":main()
