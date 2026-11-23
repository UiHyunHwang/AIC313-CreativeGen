"""Select a new Iron Man target against the Spider-Man palette and both existing neighbors."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from outputs.outward_palette16 import common as core
from outputs.outward_palette16.targets import config,open_rgb,signature,color,link_loss
from outputs.outward_thor16.portraits import HEROES

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
PRIOR=OUT/"outward_thor16"
PROMPTS=(
    "Close cinematic portrait of Iron Man in deep crimson armor, his iconic angular faceplate in brushed dark silver reflecting blue night light, narrow glowing eyes, red shoulder plates against shadowy navy steel beams.",
    "Iron Man in three-quarter view, helmet and massive red armored shoulders filling the frame, burgundy faceplate with a hint of aged copper, icy white eyes and blue steel reflections in a dark city alley.",
    "A close portrait of Iron Man in crimson and gunmetal armor, recognizable helmet with luminous narrow white eye slits, muted steel-blue light across the faceplate, deep navy industrial city at dusk.")
NEG="broad bright yellow faceplate, golden yellow lighting, white studio backdrop, unmasked human, distant figure, words"
SEEDS=range(9413000,9413006)

def palette(im,n=16):
    return signature(im,n).reshape(-1,3)
def balanced_match(a,b):
    # Sliced distribution distances account for how much of each color is required.
    from outputs.outward_palette16.targets import features
    axes=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,-1,0],[0,1,-1]],float)
    axes/=np.linalg.norm(axes,axis=1,keepdims=True)
    levels=(np.arange(128)+.5)/128
    qa=np.quantile(features(a)@axes.T,levels,axis=0)
    qb=np.quantile(features(b)@axes.T,levels,axis=0)
    return float(np.abs(qa-qb).mean())
def selection_scores(im,spider,spider_board,old_swatches,cap_patch):
    p=palette(im);r,g,b=p.T
    hero=balanced_match(p,palette(spider))
    existing=balanced_match(p,old_swatches)
    warm=float(np.maximum(0,np.minimum(r-b,g-b)-.12).mean())
    zoom_in=link_loss(im,spider_board)
    zoom_out=float(color(signature(im,4),signature(cap_patch,4)).mean())
    score=.55*hero+.15*existing+.12*zoom_in+.18*zoom_out+.30*warm
    return dict(total=score,spider_palette=hero,previous_tile_palette=existing,warm_yellow_excess=warm,
                from_spider=zoom_in,into_captain=zoom_out)
@torch.inference_mode()
def main():
    core._configure_determinism(True)
    prev=core.read_json(PRIOR/"targets.json");prev_gen=core.read_json(PRIOR/"generation.json")
    prev_edges=core.read_json(PRIOR/"render.json")["edges"]
    spider=open_rgb(OUT/prev["selected"][0]);spider_board=open_rgb(PRIOR/"spider_man_overview.png")
    raw=prev_gen["stages"][1]
    # Exact same 256 unlinked Spider-Man leaves as in the approved source.
    swatches=np.stack([signature(open_rgb(OUT/raw["directory"]/t["file"]),1)[0,0] for t in raw["tiles"]])
    cap=open_rgb(OUT/prev["selected"][2]);e=prev_edges[2];x=(e["column"]-1)*64;y=(e["row"]-1)*64
    patch=cap.crop((x,y,x+64,y+64))
    baseline=selection_scores(open_rgb(OUT/prev["selected"][1]),spider,spider_board,swatches,patch)
    d=ROOT/"portraits";d.mkdir(exist_ok=True);rows=[];pipe=None
    sheet=Image.new("RGB",(3*250,2*276),"white");draw=ImageDraw.Draw(sheet)
    for k,seed in enumerate(SEEDS):
        prompt=PROMPTS[k//2];path=d/f"iron_{seed}.png";latent=path.with_suffix(".safetensors");meta=path.with_suffix(".json")
        saved=core.read_json(meta) if meta.exists() else {}
        if not (path.exists() and latent.exists() and saved.get("prompt")==prompt and saved.get("sha256")==core.digest(path)
                and saved.get("latent")==core.digest(latent)):
            if pipe is None:pipe=core._load_pipeline(config(),True)
            z=pipe(prompt=prompt,negative_prompt=NEG,width=1024,height=1024,num_inference_steps=24,guidance_scale=5.,eta=0.,
                   generator=torch.Generator().manual_seed(seed),output_type="latent").images
            core._decode_latent(pipe,z,"cpu-tiles").save(path)
            save_file({"latents":z.cpu().contiguous()},str(latent))
            core.write_json(meta,dict(seed=seed,prompt=prompt,sha256=core.digest(path),latent=core.digest(latent)))
        im=open_rgb(path);scores=selection_scores(im,spider,spider_board,swatches,patch)
        rows.append(dict(seed=seed,file=str(path.relative_to(OUT)),prompt=prompt,**scores))
        x,y=(k%3)*250,(k//3)*276;sheet.paste(im.resize((250,250),Image.Resampling.LANCZOS),(x,y))
        draw.text((x+4,y+252),f"{seed}   score {scores['total']:.4f}",fill="black")
    sheet.save(ROOT/"candidates.jpg",quality=94)
    best=min(rows,key=lambda row:row["total"])
    paths=[OUT/p for p in prev["selected"]];paths[1]=OUT/best["file"]
    core.write_json(ROOT/"targets.json",dict(heroes=HEROES,selected=[str(p.relative_to(OUT)) for p in paths],
        hashes=[core.digest(p) for p in paths],latent_hashes=[core.digest(p.with_suffix(".safetensors")) for p in paths],
        baseline=baseline,candidates=rows,selected_seed=best["seed"],negative_prompt=NEG))
    print("baseline",baseline,"selected",best,flush=True)
if __name__=="__main__":main()
