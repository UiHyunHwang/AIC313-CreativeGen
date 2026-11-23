"""Run a separate 12-step target-conditioned SDXL trajectory for each 16x16 cell."""
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from PIL import Image
from outputs.outward_palette16.targets import ROOT,OUT,config,open_rgb,signature
from outputs.outward_palette16.common import weight
from outputs.outward_palette16 import common as exp

PROMPTS=[
    ("Captain America holding his shield","Captain America in a blue helmet, waist-up","Captain America running with his shield"),
    ("Spider-Man in his red and blue suit","Spider-Man wearing a red webbed mask, waist-up","Spider-Man crouching with his mask visible"),
    ("Iron Man in red and gold armor","Iron Man helmet, glowing eyes","Iron Man in an action pose")]
LIGHTS=("muted neutral lighting","crimson reflected light","warm copper lighting","cool steel-blue light","soft silver light")
def lighting(ref):
    r,g,b=signature(ref,1)[0,0]
    if min(r,g,b)>.58:return LIGHTS[4]
    if r-g>.13 and g-b<.065:return LIGHTS[1]
    if r-b>.13 and g-b>.04:return LIGHTS[2]
    if b-r>.045:return LIGHTS[3]
    return LIGHTS[0]

def batch_condition(pipe,prompt):
    c=exp.encode(pipe,prompt)
    return dict(encoder_hidden_states=torch.cat([c["negative_prompt_embeds"],c["prompt_embeds"]]),
                added_cond_kwargs=dict(text_embeds=torch.cat([c["negative_pooled_prompt_embeds"],c["pooled_prompt_embeds"]]),
                                      time_ids=torch.cat([c["negative_add_time_ids"],c["add_time_ids"]])))

def sample(pipe,tail,clean,ref,condition,seed):
    mask=weight(ref).cuda()
    base=torch.lerp(exp.lowpass(clean,2),clean.float(),mask).to(clean.dtype)
    noise=torch.randn(clean.shape,dtype=clean.dtype,generator=torch.Generator().manual_seed(seed)).cuda()
    z=pipe.scheduler.add_noise(base,noise,tail.times[:1])
    for k,t in enumerate(tail.times):
        both=pipe.unet(torch.cat([z,z]),t,**condition,return_dict=False)[0]
        u,v=both.chunk(2)
        z_next,x0=tail.step(u+(6.5 if k<5 else 5.)*(v-u),t,z)
        if k in exp.LATE:
            delta=clean.float()-x0.float()
            anchor=torch.lerp(exp.lowpass(delta,4),1.5*exp.lowpass(delta,16),mask)
            gain=float(tail.alphas[int(t)][1])**.5*exp.LATE[k]
            z_next=(z_next.float()+gain*anchor).to(z.dtype)
        z=z_next
    return exp._decode_latent(pipe,z)

@torch.inference_mode()
def main():
    exp._configure_determinism(True)
    selection=exp.read_json(ROOT/"targets.json")
    cfg=config()
    pipe=exp._load_pipeline(cfg,True)
    pipe.set_progress_bar_config(disable=True)
    # Encode all used combinations once while the official text encoders are resident.
    pipe.maybe_free_model_hooks()
    pipe.remove_all_hooks()
    pipe.text_encoder.to("cuda");pipe.text_encoder_2.to("cuda")
    conditions={p+", "+light:batch_condition(pipe,p+", "+light) for group in PROMPTS for p in group for light in LIGHTS}
    pipe.text_encoder.to("cpu");pipe.text_encoder_2.to("cpu")
    pipe.unet.to("cuda");pipe.vae.to("cuda")
    tail=exp.Tail(pipe.scheduler,12)
    generator_id=exp.digest(Path(__file__))
    stages=[]
    for j,(hero,filename) in enumerate(zip(selection["heroes"],selection["selected"])):
        folder=ROOT/hero;folder.mkdir(exist_ok=True)
        png=OUT/filename;latent=png.with_suffix(".safetensors")
        assert exp.digest(png)==selection["hashes"][j] and exp.digest(latent)==selection["latent_hashes"][j]
        target=open_rgb(png)
        raw=load_file(str(latent))["latents"].float().cuda()
        enlarged=F.interpolate(raw,size=(2048,2048),mode="bicubic",align_corners=False).half()
        records=[]
        for i in range(256):
            r,c=divmod(i,16);x,y=c*64,r*64
            ref=target.crop((x,y,x+64,y+64))
            prompt=PROMPTS[j][(r+2*c)%3]+", "+lighting(ref)
            seed=9130000+j*1000+i
            out=folder/f"r{r+1:02}_c{c+1:02}.png";meta=out.with_suffix(".json")
            if meta.exists() and out.exists():
                row=exp.read_json(meta)
                if row["generator"]==generator_id and row["parent"]==selection["latent_hashes"][j] and row["seed"]==seed and row["prompt"]==prompt and row["sha256"]==exp.digest(out):
                    records.append(row);continue
            clean=enlarged[...,r*128:(r+1)*128,c*128:(c+1)*128].contiguous()
            im=sample(pipe,tail,clean,ref,conditions[prompt],seed)
            tmp=out.with_suffix(".tmp.png");im.save(tmp,compress_level=2);tmp.replace(out)
            row=dict(index=i,prompt=prompt,seed=seed,generator=generator_id,parent=selection["latent_hashes"][j],
                     file=str(out.relative_to(ROOT)),sha256=exp.digest(out))
            exp.write_json(meta,row);records.append(row)
            if (i+1)%16==0:print(hero,i+1,"/256",flush=True)
        stage=dict(subject=hero,leaf=selection["heroes"][(j-1)%3],tiles=records,target=filename)
        exp.write_json(folder/"stage.json",stage);stages.append(stage)
    exp.write_json(ROOT/"generation.json",dict(grid=16,stages=stages,steps=12,native_tile=1024,
        model=cfg.model_id,revision=cfg.revision,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    print("Completed all 768 tile trajectories",flush=True)
if __name__=="__main__":main()
