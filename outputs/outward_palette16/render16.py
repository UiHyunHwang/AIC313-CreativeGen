"""Render a 16x16 SDXL mosaic cycle and evaluate the color of the zoom junctions."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
from outputs.outward_palette16 import camera
BOX=Image.Resampling.BOX
N=16
HEROES=("spider_man","iron_man","captain_america")
def xy(i,s):return (i%N*s,i//N*s)
def rgb(image):return np.asarray(image.convert("RGB"),dtype=np.float32)/255
def read(p):return json.loads(p.read_text())
def features(a):
    r,g,b=np.moveaxis(a,-1,0)
    return np.stack([.2126*r+.7152*g+.0722*b,.7*(r-g),.7*(b-g)],axis=-1)
def difference(a,b):return float(np.sqrt(np.mean((features(a)-features(b))**2,axis=-1)).mean())
def costs(template,target,child,n):
    side=1024//n; miniature=child.resize((side,side),BOX)
    # Sample the border in eight bins, reducing sensitivity to individual stroke edges.
    def stripes(im):
        return [rgb(im.crop(box).resize(shape,BOX)) for box,shape in [
            ((0,0,4,side),(1,8)),((side-4,0,side,side),(1,8)),
            ((0,0,side,4),(8,1)),((0,side-4,side,side),(8,1))]]
    border=stripes(miniature)
    pattern=rgb(child.resize((4,4),BOX))
    rows=[]
    for i in range(n*n):
        x,y=i%n*side,i//n*side
        bounds=(x,y,x+side,y+side)
        source=rgb(target.crop(bounds).resize((4,4),BOX))
        patch=rgb(template.crop(bounds).resize((4,4),BOX))
        neighbors=[]
        for k,(box,shape,ok) in enumerate([
            ((x-4,y,x,y+side),(1,8),x>0),((x+side,y,x+side+4,y+side),(1,8),x+side<1024),
            ((x,y-4,x+side,y),(8,1),y>0),((x,y+side,x+side,y+side+4),(8,1),y+side<1024)]):
            if ok:neighbors.append(difference(border[k],rgb(template.crop(box).resize(shape,BOX))))
        seam=float(np.mean(neighbors));target_fit=difference(pattern,source)
        displaced=difference(pattern,patch)
        detail=float(np.abs(np.diff(source,axis=0)).mean()+np.abs(np.diff(source,axis=1)).mean())
        score=.42*seam+.40*target_fit+.18*displaced+.08*detail
        rows.append(dict(index=i,score=score,border=seam,target=target_fit,displaced=displaced))
    return rows
def boards_and_targets():
    gen=read(ROOT/"generation.json");selection=read(ROOT/"targets.json");parents=[];boards=[]
    for j,stage in enumerate(gen["stages"]):
        assert stage["subject"]==HEROES[j] and stage["leaf"]==HEROES[j-1]
        target_file=OUT/selection["selected"][j]
        assert camera.digest(target_file)==selection["hashes"][j]
        with Image.open(target_file) as im:parents.append(im.convert("RGB"))
        board=Image.new("RGB",(16384,16384))
        assert [r["index"] for r in stage["tiles"]]==list(range(256))
        for row in stage["tiles"]:
            path=ROOT/row["file"]
            assert camera.digest(path)==row["sha256"]
            with Image.open(path) as im:
                assert im.size==(1024,1024)
                board.paste(im,xy(row["index"],1024))
        boards.append(board)
    return boards,parents
def old_results():
    folder=ROOT/"baseline"
    def load(names):
        result=[]
        for name in names:
            with Image.open(folder/name) as image:result.append(image.convert("RGB"))
        return result
    return (load([h+"_overview.png" for h in HEROES]),load([h+"_target.png" for h in HEROES]),
            read(folder/"slots.json"))
def evaluate(views,targets,slots,n):
    records={}
    for j,hero in enumerate(HEROES):
        slot=slots[j];dif=np.abs(rgb(views[j].resize((64,64),BOX))-rgb(targets[j].resize((64,64),BOX)))
        fit=costs(views[j],targets[j],views[j-1],n)[slot]
        records[hero]=dict(row=slot//n+1,column=slot%n+1,border=fit["border"],pattern=fit["target"],
                           rgb_mae64=float(dif.mean()),lower_right_mae64=float(dif[32:,32:].mean()))
    return records
def junction_gallery(old,oldslots,views,slots):
    # Diagnostic only. Dots and outlines are confined to this comparison sheet.
    sheet=Image.new("RGB",(4*230,2*260),"#f4f2ee")
    draw=ImageDraw.Draw(sheet)
    for row,j in enumerate((1,2)):
        for col,(collection,indices,n,label) in enumerate(((old,oldslots,8,"Previous"),(views,slots,16,"New"))):
            i=indices[j];r,c=divmod(i,n);side=1024//n
            left=max(0,min(c-1,n-3))*side;top=max(0,min(r-1,n-3))*side
            crop=collection[j].crop((left,top,left+3*side,top+3*side))
            full=collection[j].resize((224,224),Image.Resampling.LANCZOS)
            guide=ImageDraw.Draw(full)
            guide.rectangle((c*224//n,r*224//n,(c+1)*224//n-1,(r+1)*224//n-1),outline="#68f4d1",width=2)
            x,y=col*460,row*260
            sheet.paste(full,(x,y+24))
            sheet.paste(crop.resize((224,224),Image.Resampling.LANCZOS),(x+230,y+24))
            draw.text((x+2,y+4),f"{label} {HEROES[j]}  R{r+1} C{c+1}",fill="#222")
            draw.text((x+232,y+4),"Local 3 x 3 cells",fill="#222")
    sheet.save(ROOT/"junctions.jpg",quality=95)

def main():
    boards,targets=boards_and_targets()
    templates=[b.reduce(16) for b in boards]
    ranked=[costs(templates[j],targets[j],templates[j-1],N) for j in range(3)]
    oldv,oldt,olds=old_results()
    prior=evaluate(oldv,oldt,olds,8)
    slots=[]
    for hero,candidates in zip(HEROES,ranked):
        old=prior[hero]
        # Prefer a simultaneous improvement in surrounding border and target color.
        compatible=[v for v in candidates if v["border"]<=old["border"] and v["target"]<=old["pattern"]]
        if compatible:
            selected=min(compatible,key=lambda v:v["score"])
        else:
            def relative_cost(v):
                return max(v["border"]/max(.01,old["border"]),v["target"]/max(.01,old["pattern"]))+.25*v["score"]
            selected=min(candidates,key=relative_cost)
        slots.append(selected["index"])
    views=templates
    for _ in range(16):
        nxt=[im.copy() for im in templates]
        for j in range(3):
            nxt[j].paste(views[j-1].reduce(16),xy(slots[j],64))
        stable=all(im.tobytes()==n.tobytes() for im,n in zip(views,nxt))
        views=nxt
        if stable:break
    else:raise RuntimeError("Unstable recursive previews")
    levels=[];edges=[]
    for j,hero in enumerate(HEROES):
        pos=xy(slots[j],1024);b=boards[j]
        b.paste(views[j-1],pos)
        assert b.crop((*pos,pos[0]+1024,pos[1]+1024)).tobytes()==views[j-1].tobytes()
        assert b.reduce(16).tobytes()==views[j].tobytes()
        views[j].save(ROOT/(hero+"_overview.png"))
        b.save(ROOT/(hero+"_16x16.jpg"),quality=95,subsampling=0)
        levels.append([b.reduce(k) if k!=1 else b for k in (16,8,4,2,1)])
        edges.append(dict(outer=hero,inner=HEROES[j-1],row=slots[j]//16+1,column=slots[j]%16+1,child_sha=hashlib.sha256(views[j-1].tobytes()).hexdigest()))
    junction_gallery(oldv,olds,views,slots)
    diagnostics=dict(previous=prior,current=evaluate(views,targets,slots,16))
    sheet=Image.new("RGB",(3*300,3*325),"#fafafa");d=ImageDraw.Draw(sheet)
    for j,hero in enumerate(HEROES):
        for k,(images,label) in enumerate([(oldv,"Previous 8 x 8"),(targets,"SDXL target"),(views,"New 16 x 16")]):
            sheet.paste(images[j].resize((300,300),Image.Resampling.LANCZOS),(300*j,325*k))
            d.text((300*j+5,325*k+302),hero+"   "+label,fill="#222")
    sheet.save(ROOT/"comparison.jpg",quality=95)
    result=camera.film(levels,slots)
    (ROOT/"render.json").write_text(json.dumps(dict(grid=N,edges=edges,diagnostics=diagnostics,gif=result),indent=2)+"\n")
    print(result,diagnostics,flush=True)
if __name__=="__main__":main()
