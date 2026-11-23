"""Composition-neutral review of Spider-Man tiles. Scores only; no image generation."""
from pathlib import Path
import json
from collections import Counter
import torch
from PIL import Image,ImageDraw
from transformers import CLIPModel,CLIPProcessor
from rank_zoom import model_id,revision

import sys
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
parent=Path(json.loads((root/'selection.json').read_text())['inputs'][0])
positive=["Spider-Man in a recognizable red and blue superhero suit in an action scene","Spider-Man standing in costume, a complete figure","a portrait of Spider-Man wearing a red webbed mask with white eyes"]
negative=["a spider insect or a web without a superhero","empty scenery or a nearly uniform patch of color","a tiny unrecognizable distant speck","an unrelated person without a Spider-Man costume"]
with torch.inference_mode():
    kwargs=dict(revision=revision,local_files_only=True)
    model=CLIPModel.from_pretrained(model_id,**kwargs).eval()
    proc=CLIPProcessor.from_pretrained(model_id,use_fast=False,**kwargs)
    tok=proc(text=positive+negative,return_tensors="pt",padding=True)
    texts=torch.nn.functional.normalize(model.get_text_features(**tok))
    tiles=sorted((parent/'tiles').glob('*.png'))
    out=[]
    for start in range(0,len(tiles),8):
        batch=tiles[start:start+8]
        orig=[]
        for path in batch:
            with Image.open(path) as im:orig.append(im.convert('RGB'))
        sims=[]
        for imgs in (orig,[im.resize((32,32),Image.Resampling.BOX) for im in orig]):
            inputs=proc(images=imgs,return_tensors="pt")
            feat=torch.nn.functional.normalize(model.get_image_features(**inputs))
            sims.append(feat@texts.T)
        scores=sum(s[:,:3].amax(1)-s[:,3:].amax(1) for s in sims)/2
        neg=(sims[0]+sims[1])[:,3:].argmax(1)
        out.extend(dict(tile=p.stem,margin=float(score),nearest_contrast=negative[int(n)]) for p,score,n in zip(batch,scores,neg))
    assert len(out)==256
    visible=sorted((v for v in out if v['tile']!='r015_c014'),key=lambda v:v['margin'])
    weak=[v for v in visible if v['margin']<0]
    report=dict(positive=positive,negative=negative,model=model_id,revision=revision,locked='r015_c014',visible=len(visible),weak=len(weak),contrasts=Counter(v['nearest_contrast'] for v in weak),ranked=visible)
    (root/'spider_review.json').write_text(json.dumps(report,indent=2))
    sheet=Image.new('RGB',(4*192,2*218),'white'); draw=ImageDraw.Draw(sheet)
    for j,v in enumerate(visible[:8]):
        p=parent/'tiles'/(v['tile']+'.png');x=(j%4)*192;y=(j//4)*218
        with Image.open(p) as im:sheet.paste(im.resize((192,192),Image.Resampling.LANCZOS),(x,y))
        r,c=(int(i[1:])+1 for i in p.stem.split('_'));draw.text((x+4,y+194),f"row {r}, col {c}",fill="black")
    sheet.save(root/'spider_review.jpg',quality=92)
    print(len(weak),report['contrasts'])
