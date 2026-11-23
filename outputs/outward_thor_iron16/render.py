"""Render the revised four-stage cycle and verify all three retained parent boards."""
from pathlib import Path
import hashlib
from PIL import Image, ImageChops, ImageDraw
from outputs.outward_palette16 import camera, common as core
from outputs.outward_palette16.render16 import costs,xy
from outputs.outward_thor_iron16.targets import ROOT,OUT,PRIOR,HEROES,open_rgb

GIF="spider_iron_captain_thor_zoom_out.gif"
def render(iron):
    gen=core.read_json(ROOT/"generation.json");sel=core.read_json(ROOT/"targets.json")
    old=core.read_json(PRIOR/"render.json");original=core.read_json(PRIOR/"generation.json")
    boards=[];targets=[];slots=[]
    for j,stage in enumerate(gen["stages"]):
        h=HEROES[j];assert stage["subject"]==h and stage["leaf"]==HEROES[j-1]
        target=OUT/sel["selected"][j];assert camera.digest(target)==sel["hashes"][j]
        targets.append(open_rgb(target))
        if j==1:
            boards.append(iron);slots.append(None);continue
        assert stage==original["stages"][j]
        native=Image.new("RGB",(16384,16384))
        for i,t in enumerate(stage["tiles"]):
            file=OUT/stage["directory"]/t["file"];assert t["index"]==i and camera.digest(file)==t["sha256"]
            with Image.open(file) as im:
                assert im.size==(1024,1024)
                native.paste(im,xy(i,1024))
        boards.append(native)
        e=old["edges"][j];assert e["outer"]==h and e["inner"]==HEROES[j-1]
        slots.append((e["row"]-1)*16+e["column"]-1)
    bases=[b.reduce(16) for b in boards]
    slots[1]=min(costs(bases[1],targets[1],bases[0],16),key=lambda s:s["score"])["index"]
    views=bases
    for _ in range(16):
        nxt=[im.copy() for im in bases]
        for j,i in enumerate(slots):nxt[j].paste(views[j-1].reduce(16),xy(i,64))
        stable=all(a.tobytes()==b.tobytes() for a,b in zip(views,nxt))
        views=nxt
        if stable:break
    else:raise RuntimeError("Recursive thumbnails did not converge")
    edges=[];levels=[];kept={}
    for j,(h,b,i) in enumerate(zip(HEROES,boards,slots)):
        x,y=xy(i,1024);b.paste(views[j-1],(x,y))
        assert b.crop((x,y,x+1024,y+1024)).tobytes()==views[j-1].tobytes()
        assert b.reduce(16).tobytes()==views[j].tobytes()
        if j!=1:
            diff=ImageChops.difference(open_rgb(PRIOR/(h+"_overview.png")),views[j])
            x,y=xy(i,64);diff.paste((0,0,0),(x,y,x+64,y+64));assert diff.getbbox() is None
            kept[h]=dict(identical_native_ordinary_tiles=255,unchanged_outside_portal=True,original_slot=True)
        else:
            b.save(ROOT/"iron_man_16x16.jpg",quality=95,subsampling=0)
        views[j].save(ROOT/(h+"_overview.png"))
        levels.append([b.reduce(k) if k>1 else b for k in (16,8,4,2,1)])
        fit=costs(views[j],targets[j],views[j-1],16)[i]
        edges.append(dict(outer=h,inner=HEROES[j-1],row=i//16+1,column=i%16+1,border=fit["border"],pattern=fit["target"]))
    with Image.open(PRIOR/GIF) as f:
        palette=Image.new("P",(1,1));palette.putpalette(f.getpalette())
    gif=camera.film(levels,slots,heroes=HEROES,output_dir=ROOT,filename=GIF,palette=palette)
    assert gif["order"]==[*HEROES,HEROES[0]] and gif["duration_ms"]==40020 and gif["first_last_identical"]
    assert camera.digest(PRIOR/GIF)==old["gif"]["gif_sha256"]
    core.write_json(ROOT/"render.json",dict(gif=gif,edges=edges,preserved=kept))
    panel=Image.new("RGB",(2*360,2*386),"#f6f5f0");draw=ImageDraw.Draw(panel)
    oldsel=core.read_json(PRIOR/"targets.json")
    for j,(label,im) in enumerate((("Previous target",open_rgb(OUT/oldsel["selected"][1])),("New target",targets[1]),
                                   ("Previous mosaic",open_rgb(PRIOR/"iron_man_overview.png")),("New mosaic",views[1]))):
        x,y=j%2*360,j//2*386;panel.paste(im.resize((360,360),Image.Resampling.LANCZOS),(x,y))
        draw.text((x+6,y+363),label,fill="#222")
    panel.save(ROOT/"iron_comparison.jpg",quality=95)
    print(gif,flush=True)
