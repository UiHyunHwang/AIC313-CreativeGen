"""Render the five-board loop using the exact prior RGB leaves and new SDXL latents."""
import hashlib
import json
from io import BytesIO
from pathlib import Path
import torch
from safetensors.torch import load_file
from PIL import Image, ImageChops, ImageDraw
from outputs.outward_palette16 import common as core, camera
from outputs.outward_palette16.render16 import costs,xy
from outputs.outward_hulk16.hulk_palette import scene_palettes
from outputs.outward_hulk16.portraits import ROOT,OUT,PRIOR,HEROES,config

GIF="spider_iron_captain_thor_hulk_zoom_out.gif"
PREVIOUS_GIF="spider_iron_captain_thor_zoom_out.gif"
def read(path):return json.loads(path.read_text())

@torch.inference_mode()
def load_boards():
    data=read(ROOT/"generation.json")
    selection=read(ROOT/"targets.json")
    assert data["grid"]==16 and len(data["stages"])==5
    core._configure_determinism(True)
    pipe=core._load_pipeline(config(),True)
    pipe.maybe_free_model_hooks();pipe.remove_all_hooks()
    pipe.unet=None;pipe.text_encoder=None;pipe.text_encoder_2=None
    pipe.vae.to("cuda")
    boards=[];targets=[]
    for j,stage in enumerate(data["stages"]):
        assert stage["subject"]==HEROES[j] and stage["leaf"]==HEROES[j-1]
        path=OUT/selection["selected"][j];assert camera.digest(path)==selection["hashes"][j]
        with Image.open(path) as f:targets.append(f.convert("RGB"))
        assert [r["index"] for r in stage["tiles"]]==list(range(256))
        assert len({r["sha256"] for r in stage["tiles"]})==256
        native=Image.new("RGB",(16384,16384))
        for row in stage["tiles"]:
            p=OUT/stage["directory"]/row["file"];assert camera.digest(p)==row["sha256"]
            if p.suffix==".safetensors":
                z=load_file(str(p))["latents"].to("cuda")
                assert z.shape==(1,4,128,128) and z.dtype==torch.float16
                im=core._decode_latent(pipe,z)
                if row["index"] in (0,7,20,39,72,128,176,225):
                    check=BytesIO()
                    im.resize((240,240)).save(check,format="JPEG",quality=95)
                    assert hashlib.sha256(check.getvalue()).hexdigest()==camera.digest(p.with_suffix(".jpg"))
            else:
                with Image.open(p) as f:im=f.convert("RGB")
            assert im.size==(1024,1024)
            native.paste(im,xy(row["index"],1024))
            if (row["index"]+1)%64==0:print("Loaded",HEROES[j],row["index"]+1,flush=True)
        boards.append(native)
    return boards,targets

def main():
    boards,targets=load_boards()
    torch.cuda.empty_cache()
    templates=[b.reduce(16) for b in boards]
    previous={e["outer"]:e for e in read(PRIOR/"render.json")["edges"]}
    slots=[]
    for j,h in enumerate(HEROES):
        if j in (1,2,3):
            edge=previous[h];assert edge["inner"]==HEROES[j-1]
            slot=(edge["row"]-1)*16+edge["column"]-1
        else:
            slot=min(costs(templates[j],targets[j],templates[j-1],16),key=lambda c:c["score"])["index"]
        slots.append(slot)
    views=templates
    for _ in range(16):
        nxt=[t.copy() for t in templates]
        for j,slot in enumerate(slots):nxt[j].paste(views[j-1].reduce(16),xy(slot,64))
        stable=all(a.tobytes()==b.tobytes() for a,b in zip(views,nxt))
        views=nxt
        if stable:break
    else:raise RuntimeError("Recursive previews failed to converge")
    levels=[];edges=[];preservation={}
    sheet=Image.new("RGB",(5*288,316),"#f5f4ef");draw=ImageDraw.Draw(sheet)
    for j,(native,h,slot) in enumerate(zip(boards,HEROES,slots)):
        x,y=xy(slot,1024);native.paste(views[j-1],(x,y))
        assert native.crop((x,y,x+1024,y+1024)).tobytes()==views[j-1].tobytes()
        assert native.reduce(16).tobytes()==views[j].tobytes()
        if j in (1,2,3):
            with Image.open(PRIOR/(h+"_overview.png")) as old:
                diff=ImageChops.difference(old.convert("RGB"),views[j])
            px,py=xy(slot,64);diff.paste((0,0,0),(px,py,px+64,py+64))
            assert diff.getbbox() is None
            preservation[h]=dict(identical_ordinary_native_tiles=255,identical_overview_outside_link=True,old_slot_retained=True)
        views[j].save(ROOT/(h+"_overview.png"))
        native.save(ROOT/(h+"_16x16.jpg"),quality=95,subsampling=0)
        sheet.paste(views[j].resize((288,288),Image.Resampling.LANCZOS),(288*j,0))
        draw.text((288*j+8,294),h.replace("_"," ").title(),fill="#202020")
        levels.append([native.reduce(k) if k>1 else native for k in (16,8,4,2,1)])
        fit=costs(views[j],targets[j],views[j-1],16)[slot]
        edges.append(dict(outer=h,inner=HEROES[j-1],row=slot//16+1,column=slot%16+1,border=fit["border"],target_pattern=fit["target"],
                          child_sha=hashlib.sha256(views[j-1].tobytes()).hexdigest()))
    sheet.save(ROOT/"five_boards.jpg",quality=94)
    with Image.open(PRIOR/PREVIOUS_GIF) as old:
        approved=Image.new("P",(1,1));approved.putpalette(old.getpalette())
    palette_at,palette_info=scene_palettes(approved,views)
    gif=camera.film(levels,slots,heroes=HEROES,output_dir=ROOT,filename=GIF,palette=approved,palette_at=palette_at)
    assert gif["order"]==[*HEROES,HEROES[0]] and gif["duration_ms"]==50020
    core.write_json(ROOT/"render.json",dict(grid=16,edges=edges,preservation=preservation,gif=gif,palette=palette_info,
                                           peak_render_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    print(gif,flush=True)
if __name__=="__main__":main()
