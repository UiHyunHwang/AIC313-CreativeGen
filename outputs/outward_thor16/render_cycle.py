"""Resolve the four recursive boards and verify the actual outward GIF."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageChops, ImageDraw

from outputs.outward_palette16 import camera
from outputs.outward_palette16.render16 import costs, xy

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
PRIOR=OUT/"outward_scene16"
HEROES=("spider_man","iron_man","captain_america","thor")
GIF="spider_iron_captain_thor_zoom_out.gif"
def read(p):
    return json.loads(p.read_text())
def load_boards():
    data=read(ROOT/"generation.json")
    assert data["grid"]==16 and len(data["stages"])==4
    selection=read(ROOT/"targets.json")
    boards=[];targets=[]
    for j,stage in enumerate(data["stages"]):
        assert stage["subject"]==HEROES[j] and stage["leaf"]==HEROES[j-1]
        target=OUT/selection["selected"][j]
        assert camera.digest(target)==selection["hashes"][j]
        with Image.open(target) as f:targets.append(f.convert("RGB"))
        assert [t["index"] for t in stage["tiles"]]==list(range(256))
        assert len({t["sha256"] for t in stage["tiles"]})==256
        native=Image.new("RGB",(16384,16384))
        for tile in stage["tiles"]:
            path=OUT/stage["directory"]/tile["file"]
            assert camera.digest(path)==tile["sha256"]
            with Image.open(path) as image:
                assert image.size==(1024,1024)
                native.paste(image,xy(tile["index"],1024))
        boards.append(native)
        print("Loaded",HEROES[j],flush=True)
    return boards,targets
def main():
    boards,targets=load_boards()
    templates=[b.reduce(16) for b in boards]
    old_edges={e["outer"]:e for e in read(PRIOR/"render.json")["edges"]}
    slots=[]
    for j,h in enumerate(HEROES):
        if j in (1,2):
            edge=old_edges[h];assert edge["inner"]==HEROES[j-1]
            slot=(edge["row"]-1)*16+edge["column"]-1
        else:
            candidates=costs(templates[j],targets[j],templates[j-1],16)
            slot=min(candidates,key=lambda c:c["score"])["index"]
        slots.append(slot)
    views=templates
    for _ in range(16):
        next_views=[t.copy() for t in templates]
        for j,slot in enumerate(slots):
            next_views[j].paste(views[j-1].reduce(16),xy(slot,64))
        stable=all(a.tobytes()==b.tobytes() for a,b in zip(views,next_views))
        views=next_views
        if stable:break
    else:raise RuntimeError("Recursive previews failed to converge")
    levels=[];edges=[];preservation={}
    sheet=Image.new("RGB",(4*288,316),"#f5f4ef")
    draw=ImageDraw.Draw(sheet)
    for j,(board,h,slot) in enumerate(zip(boards,HEROES,slots)):
        x,y=xy(slot,1024)
        board.paste(views[j-1],(x,y))
        assert board.crop((x,y,x+1024,y+1024)).tobytes()==views[j-1].tobytes()
        assert board.reduce(16).tobytes()==views[j].tobytes()
        if j in (1,2):
            with Image.open(PRIOR/(h+"_overview.png")) as previous:
                diff=ImageChops.difference(previous.convert("RGB"),views[j])
            px,py=xy(slot,64)
            diff.paste((0,0,0),(px,py,px+64,py+64))
            assert diff.getbbox() is None
            preservation[h]=dict(identical_ordinary_native_tiles=255,identical_overview_outside_link=True,old_slot_retained=True)
        views[j].save(ROOT/(h+"_overview.png"))
        board.save(ROOT/(h+"_16x16.jpg"),quality=95,subsampling=0)
        sheet.paste(views[j].resize((288,288),Image.Resampling.LANCZOS),(288*j,0))
        draw.text((288*j+8,294),h.replace("_"," ").title(),fill="#202020")
        levels.append([board.reduce(k) if k>1 else board for k in (16,8,4,2,1)])
        fit=costs(views[j],targets[j],views[j-1],16)[slot]
        edges.append(dict(outer=h,inner=HEROES[j-1],row=slot//16+1,column=slot%16+1,
                          border=fit["border"],target_pattern=fit["target"],child_sha=hashlib.sha256(views[j-1].tobytes()).hexdigest()))
    sheet.save(ROOT/"four_boards.jpg",quality=94)
    with Image.open(PRIOR/"spider_iron_captain_zoom_out.gif") as old_gif:
        palette=Image.new("P",(1,1))
        palette.putpalette(old_gif.getpalette())
    gif=camera.film(levels,slots,heroes=HEROES,output_dir=ROOT,filename=GIF,palette=palette)
    assert gif["order"]==[*HEROES,HEROES[0]] and gif["duration_ms"]==40020
    report=dict(grid=16,edges=edges,preservation=preservation,gif=gif)
    (ROOT/"render.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report,flush=True)
if __name__=="__main__":main()
