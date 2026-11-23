"""Generate two new boards while keeping the four-stage source assets read-only."""
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from outputs.outward_palette16 import common as core
from outputs.outward_palette16 import generate_boards as first
from outputs.outward_scene16 import scene
from outputs.outward_thor16.new_boards import tile_spec as captain_tiles
from outputs.outward_hulk16.new_boards import denoise, tile_spec as hulk_tiles
from outputs.outward_captain_hulk_thor16.settings import ROOT, OUT, PRIOR, HEROES, config, open_rgb, select_targets

def tile_spec(hero, i, ref):
    if hero == "hulk":
        return captain_tiles("thor", i, ref)
    if hero == "thor":
        return hulk_tiles("spider_man", i, ref)
    raise ValueError(hero)

@torch.inference_mode()
def main():
    core._configure_determinism(True)
    selected=select_targets()
    old={s["subject"]:s for s in core.read_json(PRIOR/"generation.json")["stages"]}
    cfg=config()
    pipe=core._load_pipeline(cfg,True);pipe.maybe_free_model_hooks();pipe.remove_all_hooks()
    images={};prompts=set()
    for j in (3,4):
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
            flat,p=tile_spec(hero,i,ref);seed=9494000+1000*int(j==4)+i
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
