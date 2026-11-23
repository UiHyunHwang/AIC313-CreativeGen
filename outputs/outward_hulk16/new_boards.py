"""Generate two new boards while keeping the four-stage source assets read-only."""
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from outputs.outward_palette16 import common as core
from outputs.outward_palette16 import generate_boards as first
from outputs.outward_scene16 import scene
from outputs.outward_thor16.new_boards import THOR_POSES, THOR_SETS
from outputs.outward_hulk16.portraits import ROOT,OUT,PRIOR,HEROES,config,open_rgb

HULK_POSES=("Marvel's Incredible Hulk, massive green muscles and short black hair, bracing his fists",
            "the Incredible Hulk, a muscular green giant with a furrowed brow, waist-up",
            "Marvel's Hulk in torn purple trousers, crouching with his knuckles on the ground")
HULK_SETS=dict(zip(first.LIGHTS,(
    "in a graphite steel mill with heavy beams",
    "in a brick foundry reflecting crimson furnace light",
    "on a copper-lit industrial platform",
    "among storm-blue steel columns",
    "in a metal train shed with sunlit girders")))
def tile_spec(hero,i,ref):
    poses,sets=(THOR_POSES,THOR_SETS) if hero=="hulk" else (HULK_POSES,HULK_SETS)
    r,c=divmod(i,16)
    pose=poses[(r+2*c)%3]
    light=first.lighting(ref)
    flat=scene.plain(ref)
    prompt=pose+" "+sets[light]+", cinematic still, directional light and deep shadows" if flat else pose+", "+light
    return flat,prompt

def denoise(pipe,tail,clean,ref,condition,seed,flat):
    # Same two schedules as the approved run; retain the final fp16 latent for lossless decoding.
    mask=core.weight(ref).cuda()
    if flat:
        _,contrast,_=scene.properties(ref)
        mask=mask*min(1.,max(0.,(contrast-.035)/.12))
        base=torch.lerp(core.lowpass(clean,1),clean.float(),mask)*.65
    else:
        base=torch.lerp(core.lowpass(clean,2),clean.float(),mask)
    eps=torch.randn(clean.shape,dtype=clean.dtype,generator=torch.Generator().manual_seed(seed)).cuda()
    z=pipe.scheduler.add_noise(base.to(clean.dtype),eps,tail.times[:1])
    for k,t in enumerate(tail.times):
        u,v=pipe.unet(torch.cat([z,z]),t,**condition,return_dict=False)[0].chunk(2)
        next_z,x0=tail.step(u+(6.5 if k<5 else 5.)*(v-u),t,z)
        if k in core.LATE:
            delta=clean.float()-x0.float()
            shift=torch.lerp(core.lowpass(delta,1 if flat else 4),1.5*core.lowpass(delta,16),mask)
            gain=float(tail.alphas[int(t)][1])**.5*core.LATE[k]
            next_z=(next_z.float()+gain*shift).to(z.dtype)
        z=next_z
    return z

@torch.inference_mode()
def main():
    core._configure_determinism(True)
    selected=core.read_json(ROOT/"targets.json")
    old={s["subject"]:s for s in core.read_json(PRIOR/"generation.json")["stages"]}
    cfg=config()
    pipe=core._load_pipeline(cfg,True);pipe.maybe_free_model_hooks();pipe.remove_all_hooks()
    images={};prompts=set()
    for j in (0,4):
        file=OUT/selected["selected"][j];assert core.digest(file)==selected["hashes"][j]
        hero=HEROES[j];images[hero]=open_rgb(file)
        prompts.update(tile_spec(hero,i,scene.tile_ref(images[hero],i)) for i in range(256))
    pipe.text_encoder.to("cuda");pipe.text_encoder_2.to("cuda")
    conditions={(flat,p):(scene.condition if flat else first.batch_condition)(pipe,p) for flat,p in sorted(prompts)}
    pipe.text_encoder.to("cpu");pipe.text_encoder_2.to("cpu");pipe.unet.to("cuda");pipe.vae.to("cuda")
    tail=core.Tail(pipe.scheduler,12);version=core.digest(Path(__file__));stages=[]
    for j,hero in enumerate(HEROES):
        if hero not in images:
            stage=old[hero]
            assert stage["leaf"]==HEROES[j-1] and len(stage["tiles"])==256
            stages.append(stage);continue
        directory=ROOT/hero;directory.mkdir(exist_ok=True)
        filename=selected["selected"][j];latent=(OUT/filename).with_suffix(".safetensors")
        parent=core.digest(latent);assert parent==selected["latent_hashes"][j]
        raw=load_file(str(latent))["latents"].cuda().float()
        canvas=F.interpolate(raw,size=(2048,2048),mode="bicubic",align_corners=False).half()
        rows=[]
        for i in range(256):
            r,c=divmod(i,16);ref=scene.tile_ref(images[hero],i)
            flat,p=tile_spec(hero,i,ref);seed=9364000+1000*int(j==4)+i
            dest=directory/f"r{r+1:02}_c{c+1:02}.safetensors";meta=dest.with_suffix(".json")
            row=core.read_json(meta) if meta.exists() else {}
            if not (dest.is_file() and row.get("generator")==version and row.get("parent")==parent
                    and row.get("seed")==seed and row.get("prompt")==p and row.get("sha256")==core.digest(dest)):
                crop=canvas[...,r*128:(r+1)*128,c*128:(c+1)*128].contiguous()
                z=denoise(pipe,tail,crop,ref,conditions[flat,p],seed,flat)
                tmp=dest.with_suffix(".tmp");save_file({"latents":z.cpu().contiguous()},str(tmp));tmp.replace(dest)
                row=dict(index=i,file=str(dest.relative_to(ROOT)),seed=seed,prompt=p,scene=flat,
                         generator=version,parent=parent,sha256=core.digest(dest))
                core.write_json(meta,row)
                if i in (0,7,20,39,72,128,176,225):
                    core._decode_latent(pipe,z).resize((240,240)).save(dest.with_suffix(".jpg"),quality=95)
            rows.append(row)
            if (i+1)%32==0:print(hero,i+1,"/256",flush=True)
        stage=dict(subject=hero,leaf=HEROES[j-1],target=filename,directory=ROOT.name,tiles=rows)
        core.write_json(directory/"stage.json",stage);stages.append(stage)
    core.write_json(ROOT/"generation.json",dict(grid=16,stages=stages,steps=12,native_tile=1024,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,model=cfg.model_id,revision=cfg.revision))
if __name__=="__main__":main()
