"""Generate Thor in the same cinematic palette as the three existing portraits."""
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from outputs.outward_palette16 import common as core
from outputs.outward_palette16.targets import config, open_rgb, link_loss, palette_loss, signature
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
PRIOR = OUT / "outward_scene16"
HEROES = ("spider_man", "iron_man", "captain_america", "thor")
PROMPT = ("Thor, the Marvel god of thunder, a close cinematic portrait of a battle-worn blond bearded warrior "
          "with long golden hair and a deep crimson cape over dark silver Asgardian armor, "
          "Mjolnir beside his shoulder, storm-blue stone arches and warm copper rim light.")
SEEDS = (9253100, 9253101, 9253102, 9253103)

@torch.inference_mode()
def main():
    core._configure_determinism(True)
    directory=ROOT/"portraits";directory.mkdir(exist_ok=True)
    prior=core.read_json(PRIOR/"targets.json")
    spider=open_rgb(OUT/prior["selected"][0])
    captain=open_rgb(PRIOR/"captain_america_overview.png")
    palette=np.array(core.read_json(OUT/"outward_palette16/tile_palettes.json")[0]["colors"])
    sheet=Image.new("RGB",(1024,282),"white")
    draw=ImageDraw.Draw(sheet);samples=[];pipe=None
    for k,seed in enumerate(SEEDS):
        path=directory/f"thor_{seed}.png";latent=path.with_suffix(".safetensors");meta=path.with_suffix(".json")
        if not (meta.exists() and path.exists() and latent.exists() and core.read_json(meta)["png"]==core.digest(path) and core.read_json(meta)["latent"]==core.digest(latent)):
            if pipe is None:pipe=core._load_pipeline(config(),True)
            z=pipe(prompt=PROMPT,negative_prompt="white studio background, isolated cutout, distant figure",width=1024,height=1024,
                   num_inference_steps=24,guidance_scale=5.,eta=0.,generator=torch.Generator().manual_seed(seed),output_type="latent").images
            core._decode_latent(pipe,z,"cpu-tiles").save(path)
            save_file({"latents":z.cpu().contiguous()},str(latent))
            core.write_json(meta,dict(seed=seed,prompt=PROMPT,png=core.digest(path),latent=core.digest(latent)))
        im=open_rgb(path)
        costs=[link_loss(im,captain),link_loss(spider,im),palette_loss(im,palette)]
        samples.append(dict(seed=seed,file=str(path.relative_to(OUT)),costs=costs,score=sum(costs)))
        sheet.paste(im.resize((256,256),Image.Resampling.LANCZOS),(256*k,0))
        draw.text((256*k+6,259),str(seed),fill="black")
    sheet.save(ROOT/"thor_candidates.jpg",quality=94)
    best=min(samples,key=lambda s:s["score"])
    chosen=OUT/best["file"]
    core.write_json(ROOT/"targets.json",dict(heroes=HEROES,selected=[*prior["selected"],best["file"]],
        hashes=[*prior["hashes"],core.digest(chosen)],latent_hashes=[*prior["latent_hashes"],core.digest(chosen.with_suffix(".safetensors"))],
        candidates=samples,seed=best["seed"],prompt=PROMPT))
    print("Selected",best,flush=True)
if __name__=="__main__":main()
