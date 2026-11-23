"""Compare four hypothetical Hulk insertions. This analysis writes no animation or board."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
HEROES=("spider_man","iron_man","captain_america","thor")
def read(p):return json.loads(p.read_text())
def pixels(im,n):return np.asarray(im.convert("RGB").resize((n,n),Image.Resampling.BOX),dtype=float)/255
def lab(a):
    # Björn Ottosson's Oklab definition, multiplied by 100 for display.
    a=np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
    lms=a@np.array([[.4122214708,.5363325363,.0514459929],[.2119034982,.6806995451,.1073969566],[.0883024619,.2817188376,.6299787005]]).T
    return 100*np.cbrt(lms)@np.array([[.2104542553,.7936177850,-.0040720468],[1.9779984951,-2.4285922050,.4505937099],[.0259040371,.7827717662,-.8086757660]]).T

def terms(child,parent):
    c=lab(pixels(child,16)).reshape(-1,3);p=lab(pixels(parent,16)).reshape(-1,3)
    # Emphasize chroma here, because the question concerns hue compatibility.
    weights=np.array([.5,1,1])
    c*=weights;p*=weights
    distances=np.linalg.norm(p[:,None]-c[None,:],axis=-1)
    coverage=float(np.partition(distances,3,axis=1)[:,:4].mean())
    # The same number of patches on each side also allows quantile comparison.
    axes=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,-1,0],[0,1,-1]],float)
    axes/=np.linalg.norm(axes,axis=1,keepdims=True)
    distribution=float(np.abs(np.sort(c@axes.T,axis=0)-np.sort(p@axes.T,axis=0)).mean())
    field=lab(pixels(parent,64)).reshape(16,4,16,4,3).transpose(0,2,1,3,4)
    child_pattern=lab(pixels(child,4))
    best_portal=float(np.linalg.norm((field-child_pattern)*weights,axis=-1).mean(axis=(2,3)).min())
    total=.4*coverage+.4*distribution+.2*best_portal
    return dict(score=total,palette_coverage=coverage,distribution=distribution,best_portal=best_portal)
def ranking(images):
    rows=[]
    for i,prev in enumerate(HEROES):
        nxt=HEROES[(i+1)%4]
        a=terms(images[prev],images["hulk"]);b=terms(images["hulk"],images[nxt])
        rows.append(dict(predecessor=prev,successor=nxt,incoming=a,outgoing=b,mean=(a["score"]+b["score"])/2))
    return sorted(rows,key=lambda row:row["mean"])
def main():
    target=read(ROOT/"targets.json");h=read(OUT/"outward_hulk16/targets.json")
    filepaths=dict(zip(HEROES,target["selected"]));filepaths["hulk"]=h["selected"][4]
    images={k:Image.open(OUT/v).convert("RGB") for k,v in filepaths.items()}
    rows=ranking(images)
    overviews={k:Image.open(ROOT/(k+"_overview.png")).convert("RGB") for k in HEROES}
    overviews["hulk"]=Image.open(OUT/"outward_hulk16/hulk_overview.png").convert("RGB")
    check=ranking(overviews)
    centers={}
    for fraction in (.5,.67,.75):
        crops={}
        for hero,im in images.items():
            dx=int(im.width*(1-fraction)/2);dy=int(im.height*(1-fraction)/2)
            crops[hero]=im.crop((dx,dy,im.width-dx,im.height-dy))
        centers[str(fraction)]=ranking(crops)
    result=dict(target_files=filepaths,method="Directed palette coverage (40%), projected color distributions (40%), best local 4x4 match (20%); equal weight on the two transitions. Distances are Oklab with half lightness weight.",
                primary=rows,completed_overview_sensitivity=check,central_crop_sensitivity=centers,
                scope="Existing four-stage order. Existing five-stage Hulk supplies the reference portrait. Estimates only; no candidate insertion rendered.")
    (ROOT/"hulk_placement.json").write_text(json.dumps(result,indent=2)+"\n")
    print([(v["predecessor"],v["successor"],round(v["mean"],3)) for v in rows])
if __name__=="__main__":main()
