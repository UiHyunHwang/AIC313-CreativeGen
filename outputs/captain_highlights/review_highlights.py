"""Evaluate the character and highlight criteria on the same complete board."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
FIRST=ROOT.parent/"captain_firstpass"
OLD=ROOT.parent/"outward_hero_cycle"
sys.path.insert(0,str(FIRST))
import review

def light(image,side):
    return np.asarray(image.convert("RGB").resize((side,side),Image.Resampling.BOX),dtype=float)/255 @ [.2126,.7152,.0722]

def raw_board(directory,tiles):
    canvas=np.zeros((64,64))
    for tile in tiles:
        i=tile["index"]
        y,x=8*(i//8),8*(i%8)
        with Image.open(directory/tile["file"]) as im:
            canvas[y:y+8,x:x+8]=light(im,8)
    return canvas

def diagnostic(boards,target):
    bright=target>.7
    return {key:dict(highlight=float(image[bright].mean()),luma64_mae=float(abs(image-target).mean()))
            for key,image in boards.items()}

def passes(values,gate):
    original=values["baseline"]
    first=values["first"]
    new=values["candidate"]
    return dict(highlight=new["highlight"] >= first["highlight"]+gate["min_highlight_recovery"]*
        (original["highlight"]-first["highlight"]),
        luma64=new["luma64_mae"]<=original["luma64_mae"]+gate["max_raw_board_luma64_extra"])

def main():
    stage=review.read(OLD/"spider_man/stage.json")
    rows=[review.read(p) for p in sorted((ROOT/"spider_man").glob("r*_c*.json"))]
    assert [r["index"] for r in rows]==list(range(64))
    with Image.open(ROOT.parent/"spider_iron_captain"/stage["source"]) as im:
        target=light(im,64)
    dirs={"baseline":OLD,"first":FIRST,"candidate":ROOT}
    values=diagnostic({k:raw_board(d,stage["tiles"]) for k,d in dirs.items()},target)
    gate=review.read(ROOT/"protocol.json")["secondary_gate"]
    checks=passes(values,gate)
    result=dict(raw_board=values,checks=checks,active_tiles=sum(r["active"] for r in rows),highlight_cells=int((target>.7).sum()))
    review.write(ROOT/"highlight_comparison.json",result)
    print(json.dumps(result),flush=True)
    review.ROOT=ROOT
    sys.argv=sys.argv[:1]+(["--render"] if all(checks.values()) else [])
    review.main()
    if (ROOT/"render.json").exists():
        boards={}
        for k,d in dirs.items():
            with Image.open(d/"spider_man_overview.png") as im:
                boards[k]=light(im,64)
        after=diagnostic(boards,target)
        result.update(rendered_board=after,rendered_checks=passes(after,gate))
        review.write(ROOT/"highlight_comparison.json",result)
        print(json.dumps(result["rendered_checks"]),flush=True)

if __name__=="__main__":
    main()
