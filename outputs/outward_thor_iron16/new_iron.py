"""Keep the approved scene sampler and coordinate seeds; replace Iron Man's target latent."""
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from safetensors.torch import load_file,save_file
from outputs.outward_palette16 import common as core
from outputs.outward_scene16 import scene
from outputs.outward_thor_iron16.targets import ROOT, OUT, PRIOR, config, open_rgb

def denoise(pipe,tail,clean,ref,cond,seed):
    # Return the scene sampler's fp16 final state; this compact record can be decoded losslessly.
    _,contrast,_=scene.properties(ref)
    flat=scene.plain(ref);contour=min(1.,max(0.,(contrast-.035)/.12))
    mask=core.weight(ref).cuda()*contour
    coarse=core.lowpass(clean,1 if flat else 2)
    base=torch.lerp(coarse,clean.float(),mask)*(.65 if flat else .88)
    eps=torch.randn(clean.shape,dtype=clean.dtype,generator=torch.Generator().manual_seed(seed)).cuda()
    z=pipe.scheduler.add_noise(base.to(clean.dtype),eps,tail.times[:1])
    for k,t in enumerate(tail.times):
        u,v=pipe.unet(torch.cat([z,z]),t,**cond,return_dict=False)[0].chunk(2)
        nxt,x0=tail.step(u+(6.5 if k<5 else 5.)*(v-u),t,z)
        if k in core.LATE:
            delta=clean.float()-x0.float()
            shift=torch.lerp(core.lowpass(delta,1 if flat else 4),1.5*core.lowpass(delta,16),mask)
            gain=float(tail.alphas[int(t)][1])**.5*core.LATE[k]
            nxt=(nxt.float()+gain*shift).to(z.dtype)
        z=nxt
    return z

@torch.inference_mode()
def generate():
    sel=core.read_json(ROOT/"targets.json")
    old=core.read_json(PRIOR/"generation.json")
    source=old["stages"][1]
    file=OUT/sel["selected"][1];latent=file.with_suffix(".safetensors")
    parent=core.digest(latent)
    assert core.digest(file)==sel["hashes"][1] and parent==sel["latent_hashes"][1]
    target=open_rgb(file);cfg=config();core._configure_determinism(True)
    pipe=core._load_pipeline(cfg,True);pipe.maybe_free_model_hooks();pipe.remove_all_hooks()
    pipe.text_encoder.to("cuda");pipe.text_encoder_2.to("cuda")
    prompts=[scene.make_prompt(1,i,scene.tile_ref(target,i)) for i in range(256)]
    conds={p:scene.condition(pipe,p) for p in sorted(set(prompts))}
    pipe.text_encoder.to("cpu");pipe.text_encoder_2.to("cpu");pipe.unet.to("cuda");pipe.vae.to("cuda")
    tail=core.Tail(pipe.scheduler,12);raw=load_file(str(latent))["latents"].cuda().float()
    canvas=F.interpolate(raw,size=(2048,2048),mode="bicubic",align_corners=False).half()
    board=Image.new("RGB",(16384,16384));rows=[];version=core.digest(Path(__file__))
    directory=ROOT/"iron_man";directory.mkdir(exist_ok=True)
    for i,original in enumerate(source["tiles"]):
        assert original["index"]==i
        r,c=divmod(i,16);seed=original["seed"];p=prompts[i];ref=scene.tile_ref(target,i)
        dest=directory/f"r{r+1:02}_c{c+1:02}.safetensors";meta=dest.with_suffix(".json")
        row=core.read_json(meta) if meta.exists() else {}
        if dest.exists() and row.get("generator")==version and row.get("seed")==seed and row.get("parent")==parent and row.get("prompt")==p and row.get("sha256")==core.digest(dest):
            z=load_file(str(dest))["latents"].cuda()
        else:
            crop=canvas[...,r*128:(r+1)*128,c*128:(c+1)*128].contiguous()
            z=denoise(pipe,tail,crop,ref,conds[p],seed)
            tmp=dest.with_suffix(".tmp");save_file({"latents":z.cpu().contiguous()},str(tmp));tmp.replace(dest)
            row=dict(index=i,file=str(dest.relative_to(ROOT)),seed=seed,prompt=p,parent=parent,generator=version,sha256=core.digest(dest))
            core.write_json(meta,row)
        im=core._decode_latent(pipe,z)
        board.paste(im,(c*1024,r*1024));rows.append(row)
        if i in (0,7,20,39,72,128,176,225):
            im.resize((220,220),Image.Resampling.LANCZOS).save(dest.with_suffix(".jpg"),quality=95)
        if (i+1)%32==0:print("Spider-Man tiles",i+1,"/256",flush=True)
    stage=dict(subject="iron_man",leaf="spider_man",target=sel["selected"][1],directory=ROOT.name,tiles=rows)
    stages=old["stages"].copy();stages[1]=stage
    core.write_json(directory/"stage.json",stage)
    core.write_json(ROOT/"generation.json",dict(grid=16,native_tile=1024,steps=12,stages=stages,model=cfg.model_id,
                                               revision=cfg.revision,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    return board
if __name__=="__main__":
    from outputs.outward_thor_iron16.render import render
    board=generate()
    torch.cuda.empty_cache()
    render(board)
