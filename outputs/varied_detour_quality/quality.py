"""Choose only from SDXL samples previously generated at the same target and grid position."""
import json
from pathlib import Path
import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor,CLIPModel
from rank_zoom import model_id,revision
from sdxl.hero_cycle_render import cell,BOX,read,digest

def apply_quality(root,base,runs):
    archive=root.parent/'hero_cycle'
    cap=root.parent/'captain_refined'
    with Image.open(runs[0]/'target.png') as im:target=im.convert('RGB')
    assert digest(archive/'opening_target.png')==digest(runs[0]/'target.png')
    cap_parent=cap/read(cap/'report.json')['raw_runs'][1]
    assert digest(cap_parent/'final_latents.safetensors')==digest(runs[1]/'final_latents.safetensors')
    review=read(root/'spider_review.json')
    rows=review['ranked']
    # Limit changes to a quarter of the outer board to preserve variety.
    weak={int(v['tile'][1:4])*16+int(v['tile'][6:]):v for v in rows[:64] if v['margin']<0}
    pool=[]
    for f in sorted((archive/'spider/samples').glob('*.json')):
        v=read(f)
        if v['index'] in weak:pool.append(v)
    opts=dict(revision=revision,local_files_only=True)
    processor=CLIPProcessor.from_pretrained(model_id,use_fast=False,**opts)
    model=CLIPModel.from_pretrained(model_id,**opts).eval()
    pos,neg=review['positive'],review['negative']
    def small(im):return np.asarray(im.resize((4,4),BOX),dtype=float)/255
    refs={i:small(cell(target,i)) for i in weak}
    old_errors={}
    for i,r in weak.items():
        with Image.open(runs[0]/'tiles'/(r['tile']+'.png')) as im:old_errors[i]=float(np.abs(small(im)-refs[i]).mean())
    choices={};scores=[]
    with torch.inference_mode():
        text=torch.nn.functional.normalize(model.get_text_features(**processor(text=pos+neg,return_tensors='pt',padding=True)))
        for start in range(0,len(pool),8):
            rows_=pool[start:start+8]
            imgs=[]
            for v in rows_:
                with Image.open((archive/v['file']).with_suffix('.png')) as im:imgs.append(im.convert('RGB'))
            margins=[]
            for views in (imgs,[im.resize((32,32),BOX) for im in imgs]):
                features=torch.nn.functional.normalize(model.get_image_features(**processor(images=views,return_tensors='pt')))
                sim=features@text.T;margins.append(sim[:,:len(pos)].amax(1)-sim[:,len(pos):].amax(1))
            for v,im,score in zip(rows_,imgs,sum(margins)/2):
                i=v['index'];e=float(np.abs(refs[i]-small(im)).mean());val=float(score)
                rec=dict(index=i,file=v['file'],prompt=v['prompt'],seed=v['seed'],steps=v['steps'],margin=val,rgb=e,old_margin=weak[i]['margin'],old_rgb=old_errors[i])
                scores.append(rec)
                if val<weak[i]['margin']+.012 or e>old_errors[i]+.025:continue
                if i not in choices or val-.2*e>choices[i]['margin']-.2*choices[i]['rgb']:choices[i]=rec
    c=read(cap/'refinement.json')
    overrides=[{i:(archive/v['file']).with_suffix('.png') for i,v in choices.items()},
               {v['index']:(cap/v['file']).with_suffix('.png') for v in c['selected'] if v['seed']!=43},{},{}]
    assert 254 not in overrides[0] and 173 not in overrides[1]
    quality=dict(spider_eligible=len(weak),spider_candidates=len(scores),spider_changed=len(choices),captain_changed=len(overrides[1]),
        spider_before_negative=review['weak'],spider_after_negative=review['weak']-sum(v['margin']>=0 for v in choices.values()),
        captain_before=c['before'],captain_after=c['after'],selected=list(choices.values()),candidates=scores)
    (root/'quality.json').write_text(json.dumps(quality,indent=2))
    print("quality",len(choices),len(overrides[1]),flush=True)
    return overrides
