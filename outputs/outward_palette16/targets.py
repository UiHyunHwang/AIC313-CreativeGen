"""Create and select two new SDXL portraits against the existing tile palettes."""
from pathlib import Path
from itertools import product
from dataclasses import replace
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import save_file
from outputs.outward_palette16 import common as exp

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
BOX=Image.Resampling.BOX
SPIDER=ROOT/"opening.png"
PROMPTS={
    "iron_man":[
        "Iron Man's unmistakable helmet filling the frame, front-facing, deep lacquer-red armor and a brushed champagne faceplate reflecting burgundy and slate-blue light, charcoal background, cinematic portrait.",
        "A close portrait of Iron Man's red helmet and glowing ivory eyes, muted copper faceplate, cool steel-blue reflections along the jaw against a dark crimson industrial background."],
    "captain_america":[
        "Captain America in his navy blue A helmet, a large clear face above the curved red and silver rim of his shield, warm red studio lighting over one shoulder and gunmetal shadows, cinematic close portrait.",
        "A centered close portrait of Captain America wearing his dark blue A helmet, his face in warm copper light with a broad crimson shield and charcoal steel background."]}
HEROES=("spider_man","iron_man","captain_america")
def config():
    return replace(exp.RunConfig.read(ROOT/"recipe.json"),offload="model",gpu_memory_limit_gib=None,min_free_gpu_gib=0.)
def signature(im,n):
    return np.asarray(im.convert("RGB").resize((n,n),BOX),dtype=np.float32)/255
def open_rgb(p):
    with Image.open(p) as im:
        return im.convert("RGB")
def swatches(im):
    return signature(im,16).reshape(256,3)
def features(x):
    r,g,b=np.moveaxis(x,-1,0)
    return np.stack([.2126*r+.7152*g+.0722*b, .70*(r-g), .70*(b-g)],axis=-1)
def color(x,y):
    return np.sqrt(np.mean((features(x)-features(y))**2,axis=-1))
def palette_loss(im,palette):
    d=color(swatches(im)[:,None,:],palette[None,:,:])
    return float(d.min(1).mean())
def link_loss(parent,child):
    # Compare the child's full 4x4 color pattern to each candidate 64-pixel patch.
    fields=signature(parent,64).reshape(16,4,16,4,3).transpose(0,2,1,3,4)
    costs=color(fields,signature(child,4)).mean((2,3)).ravel()
    return float(costs.min())
def previous_tiles():
    folders=[OUT/"captain_highlights/spider_man",OUT/"dualscale_reveal/tiles",OUT/"outward_hero_cycle/captain_america"]
    path=ROOT/"tile_palettes.json"
    if path.exists():
        return [np.array(group["colors"],dtype=np.float32) for group in exp.read_json(path)]
    groups=[]
    for folder in folders:
        paths=sorted(folder.glob("r*_c*.png"))
        assert len(paths)==64
        groups.append(dict(files=[str(p.relative_to(OUT)) for p in paths],hashes=[exp.digest(p) for p in paths],
                           colors=[signature(open_rgb(p),1)[0,0].tolist() for p in paths]))
    exp.write_json(path,groups)
    return [np.array(group["colors"],dtype=np.float32) for group in groups]
@torch.inference_mode()
def main():
    exp._configure_determinism(True)
    cfg=config()
    pipe=None
    dirs=ROOT/"targets";dirs.mkdir(exist_ok=True)
    groups={}
    for j,(hero,prompts) in enumerate(PROMPTS.items()):
        groups[hero]=[]
        for i in range(4):
            prompt=prompts[i//2]; seed=9108200+10*j+i
            name=dirs/f"{hero}_{i}";png=name.with_suffix(".png");latent=name.with_suffix(".safetensors");meta=name.with_suffix(".json")
            if not (meta.exists() and png.exists() and latent.exists() and exp.read_json(meta)["sha256"]==exp.digest(png) and exp.read_json(meta)["latent_sha"]==exp.digest(latent)):
                if pipe is None:pipe=exp._load_pipeline(cfg,True)
                z=pipe(prompt=prompt,negative_prompt="",width=1024,height=1024,num_inference_steps=24,guidance_scale=5.,
                       eta=0.,generator=torch.Generator().manual_seed(seed),output_type="latent").images
                im=exp._decode_latent(pipe,z,"cpu-tiles")
                im.save(png)
                save_file({"latents":z.cpu().contiguous()},str(latent))
                exp.write_json(meta,dict(seed=seed,prompt=prompt,sha256=exp.digest(png),latent_sha=exp.digest(latent)))
            groups[hero].append(open_rgb(png))
            print(hero,i,flush=True)
    spider=open_rgb(SPIDER)
    palettes=previous_tiles()
    pairs=[]
    for i,c in product(range(4),repeat=2):
        iron=groups["iron_man"][i];cap=groups["captain_america"][c]
        pl=[palette_loss(iron,palettes[1]),palette_loss(cap,palettes[2])]
        links=[link_loss(iron,spider),link_loss(cap,iron),link_loss(spider,cap)]
        pairs.append(dict(iron=i,captain=c,palette=pl,links=links,score=float(np.mean(pl)+np.mean(links)+.35*max(links))))
    eligible={"iron":[0,3],"captain":[2,3]}  # Visual shortlist: large readable helmets before generating the boards.
    best=min((p for p in pairs if p["iron"] in eligible["iron"] and p["captain"] in eligible["captain"]),key=lambda a:a["score"])
    paths=[SPIDER,dirs/f'iron_man_{best["iron"]}.png',dirs/f'captain_america_{best["captain"]}.png']
    exp.write_json(ROOT/"targets.json",dict(grid=16,heroes=HEROES,selected=[str(p.relative_to(OUT)) for p in paths],
        hashes=[exp.digest(p) for p in paths],latent_hashes=[exp.digest(p.with_suffix(".safetensors")) for p in paths],
        portrait_shortlist=eligible,pairs=pairs,best=best,model=cfg.model_id,revision=cfg.revision))
    contact=Image.new("RGB",(4*210,2*234),"white"); draw=ImageDraw.Draw(contact)
    for j,hero in enumerate(PROMPTS):
        for i,im in enumerate(groups[hero]):
            contact.paste(im.resize((210,210)),(i*210,j*234))
            selected=(best["iron" if j==0 else "captain"]==i)
            draw.text((i*210+3,j*234+211),f"{hero} {i}"+("   selected" if selected else ""),fill="black")
    contact.save(ROOT/"target_candidates.jpg",quality=92)
    print(best,flush=True)
if __name__=="__main__":main()
