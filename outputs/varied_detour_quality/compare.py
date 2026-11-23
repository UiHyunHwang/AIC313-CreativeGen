"""Show the four weakest original leaves in each layer, before and after selection."""
from pathlib import Path
from PIL import Image,ImageDraw
import json
import sys

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
spider=read(root/"spider_review.json")["ranked"]
baseline=read(root.parent/"captain_refined/baseline.json")
captain=sorted((v for v in baseline["tiles"] if v["index"]!=173),key=lambda v:(v["clip"]+v["variety"],v["index"]))
updated_spider={r["index"]:r for r in read(root/"quality.json")["selected"]}
updated_cap={r["index"]:r for r in read(root.parent/"captain_refined/refinement.json")["selected"]}
parents=[Path(p) for p in read(root/"selection.json")["inputs"]]
sheet=Image.new("RGB",(4*180,4*205),"white")
pen=ImageDraw.Draw(sheet)
for stage,entries in enumerate([spider,captain]):
    for col,r in enumerate(entries[:4]):
        i=r["index"] if stage else int(r["tile"][1:4])*16+int(r["tile"][6:])
        row,c=divmod(i,16)
        src=parents[stage]/"tiles"/f"r{row:03}_c{c:03}.png"
        if stage:
            new=(root.parent/"captain_refined"/updated_cap[i]["file"]).with_suffix(".png")
        elif i in updated_spider:
            new=(root.parent/"hero_cycle"/updated_spider[i]["file"]).with_suffix(".png")
        else:new=src
        for k,path in enumerate((src,new)):
            x=180*col;y=(stage*2+k)*205
            pen.text((x+3,y+4),f"{'Captain' if stage else 'Spider'}  {'After' if k else 'Before'}  ({row+1},{c+1})",fill="black")
            with Image.open(path) as im:sheet.paste(im.resize((180,180),Image.Resampling.LANCZOS),(x,y+23))
sheet.save(root/"tile_comparison.jpg",quality=94)
