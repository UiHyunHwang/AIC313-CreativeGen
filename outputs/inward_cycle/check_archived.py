"""Use a previously sampled same-coordinate SDXL alternative only if both views clearly identify the character."""
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from refine_spider import Judge, DEFAULT
from sdxl.hero_cycle import patch,read_json,write_json

@torch.inference_mode()
def evaluate(root,prior):
    archive=root.parent/"hero_cycle"
    base=root.parent/"hero_cycle_varied"
    raw=base/read_json(base/"report.json")["raw_runs"][0]
    review=read_json(prior/"spider_review.json")
    judge=Judge(review)
    with Image.open(raw/"target.png") as im: target=im.convert("RGB")
    rows=[]
    for src in sorted((archive/"spider/samples").glob("*.json")):
        rec=read_json(src)
        if rec["index"]==254:continue
        path=(archive/rec["file"]).with_suffix(".png")
        with Image.open(path) as im:
            metrics=judge.score("spider",im,patch(target,rec["index"]))
        rows.append({k:rec[k] for k in ("index","prompt","steps","seed","file")}|metrics)
    write_json(root/"archived_scores.json",rows)

def select(root,prior):
    archive=root.parent/"hero_cycle"
    base=root.parent/"hero_cycle_varied"
    raw=base/read_json(base/"report.json")["raw_runs"][0]
    earlier={r["index"]:r for r in read_json(prior/"quality.json")["selected"]}
    records=read_json(root/"spider.json")
    baseline={r["index"]:r for r in records["baseline"]}
    fresh={r["index"]:r for r in records["selected"]}
    current=baseline|fresh
    best={}
    for c in read_json(root/"archived_scores.json"):
        i=c["index"];original=current[i]
        if min(original["native"],original["small"])>=-.01 and original["margin"]>=0:continue
        if min(c["native"],c["small"])<=.005 or c["margin"]<original["margin"]+.012:continue
        if c["error"]>min(original["error"]+.015,baseline[i]["error"]+.03):continue
        if i not in best or c["margin"]-.24*c["error"]>best[i]["margin"]-.24*best[i]["error"]:
            best[i]=c
    final=current|best
    def stats(rows):
        return dict(average_negative=sum(v["margin"]<0 for v in rows.values()),either_negative=sum(min(v["native"],v["small"])<0 for v in rows.values()))
    write_json(root/"final_selection.json",dict(fresh_used=len(fresh.keys()-best.keys()),
        archive_used=len(best),changes=len(fresh.keys()|best.keys()),baseline=stats(baseline),final=stats(final),
        selected=list(best.values()),reviewed_archive=len(read_json(root/"archived_scores.json"))))
    sheet=Image.new("RGB",(4*190,4*210),"white")
    draw=ImageDraw.Draw(sheet)
    for j,old in enumerate(sorted(baseline.values(),key=lambda v:(v["margin"],v["index"]))[:8]):
        i=old["index"]
        src=(archive/earlier[i]["file"]).with_suffix(".png") if i in earlier else raw/"tiles"/f"r{i//16:03}_c{i%16:03}.png"
        dst=(archive/best[i]["file"]).with_suffix(".png") if i in best else (root/fresh[i]["file"]).with_suffix(".png") if i in fresh else src
        for k,img in enumerate((src,dst)):
            x=190*(j%4);y=210*(2*(j//4)+k)
            with Image.open(img) as im:sheet.paste(im.resize((190,190),Image.Resampling.LANCZOS),(x,y+20))
            draw.text((x+4,y+2),f"{'After' if k else 'Before'}  ({i//16+1},{i%16+1})",fill="black")
    sheet.save(root/"spider_comparison.jpg",quality=94)
    print("Visible leaf stats",stats(baseline),stats(final),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("evaluate","select"))
    parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--prior",type=Path,default=Path("outputs/varied_detour_quality"))
    args=parser.parse_args()
    (evaluate if args.stage=="evaluate" else select)(args.output,args.prior)
