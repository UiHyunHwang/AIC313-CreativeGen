"""Evaluate a whole-board generation policy and render it if its paired gate passes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "outward_hero_cycle"
sys.path.insert(0, str(OLD))
BOX = Image.Resampling.BOX
LANCZOS = Image.Resampling.LANCZOS

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

def summarize(rows, gate):
    summary = {}
    for metric in rows[0]["baseline"]:
        old, new = np.array([[row[mode][metric] for row in rows]
                             for mode in ("baseline", "candidate")])
        change = new - old
        rng = np.random.default_rng(306)
        boots = rng.choice(change, (10000, len(change))).mean(1)
        summary[metric] = dict(old=float(old.mean()), new=float(new.mean()),
            delta=float(change.mean()), increases=int((change > 0).sum()),
            old_negative=int((old < 0).sum()), new_negative=int((new < 0).sum()),
            paired_bootstrap_95=np.quantile(boots,[.025,.975]).tolist())
    conditions = dict(
        gain88=summary["margin88"]["delta"] >= gate["min_gain88"],
        majority=summary["margin88"]["increases"] > len(rows)*gate["min_win_fraction"],
        native=summary["native"]["delta"] >= -gate["max_native_loss"],
        luma=summary["luma"]["delta"] <= gate["max_luma_increase"],
        rgb=summary["rgb"]["delta"] <= gate["max_rgb_increase"])
    return dict(n=len(rows), metrics=summary, checks=conditions, passed=all(conditions.values()))

def load_image(path, side):
    with Image.open(path) as im:
        return im.convert("RGB").resize((side, side), BOX)

def galleries(rows):
    boards = [Image.new("RGB", (704,704)) for _ in range(2)]
    for row in rows:
        i=row["index"]
        for board, folder in zip(boards, (OLD, ROOT)):
            view=load_image(folder/row["file"],128).resize((88,88),LANCZOS)
            board.paste(view, (88*(i%8), 88*(i//8)))
    comparison=Image.new("RGB",(1424,735),"white")
    draw=ImageDraw.Draw(comparison)
    for j,(label,board) in enumerate(zip(("Previous generation", "Subject first, late tone guidance"),boards)):
        draw.text((8+720*j,8),label,fill="black")
        comparison.paste(board,(720*j,30))
    comparison.save(ROOT/"board_comparison.jpg",quality=96)
    # Every tile in scan order, with paired columns. Mark the recursive portal.
    portal=read(ROOT/"protocol.json")["excluded_portal"]
    album=Image.new("RGB",(4*232,16*133),"#faf9f4")
    draw=ImageDraw.Draw(album)
    for i,row in enumerate(rows):
        x,y=i%4*232,i//4*133
        draw.text((x+3,y+2),f"R{i//8+1} C{i%8+1}  old / new"+(" *" if i==portal else ""),fill="#222222")
        for j,folder in enumerate((OLD,ROOT)):
            album.paste(load_image(folder/row["file"],110),(x+112*j+2,y+18))
    album.save(ROOT/"all_pairs.jpg",quality=95)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--render",action="store_true")
    args=parser.parse_args()
    protocol=read(ROOT/"protocol.json")
    rows=[read(p) for p in sorted((ROOT/"spider_man").glob("r*_c*.json"))]
    assert [r["index"] for r in rows] == list(range(64))
    assert all(r["generator"]==protocol["generator_sha256"] and
               hashlib.sha256((ROOT/r["file"]).read_bytes()).hexdigest()==r["sha256"] for r in rows)
    ordinary = [r for r in rows if r["index"]!=protocol["excluded_portal"]]
    result=summarize(ordinary,protocol["gate"])
    result["by_prompt"] = {p:summarize([r for r in ordinary if r["prompt"]==p],protocol["gate"])["metrics"]["margin88"]
                           for p in sorted({r["prompt"] for r in rows})}
    result["all_64"]=summarize(rows,protocol["gate"])
    galleries(rows)
    write(ROOT/"comparison.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("by_prompt","all_64")}),flush=True)
    if args.render and result["passed"]:
        import render_outward as render
        boards,targets=render.inputs()
        for row in rows:
            i=row["index"]
            with Image.open(ROOT/row["file"]) as im:
                boards[0].paste(im,(1024*(i%8),1024*(i//8)))
        render.ROOT=ROOT
        levels,slots,costs,edges=render.resolve(boards,targets)
        visible=summarize([r for r in rows if r["index"]!=slots[0]],protocol["gate"])
        write(ROOT/"visible_comparison.json",visible)
        if not visible["passed"]:
            print("New portal changes the gate. Preserving previous film.",flush=True)
            return
        gif=render.film(levels,slots)
        write(ROOT/"render.json",dict(gif=gif,edges=edges,slot_costs=costs))
        print(json.dumps(gif),flush=True)

if __name__=="__main__":
    main()
