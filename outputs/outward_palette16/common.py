from dataclasses import replace
from pathlib import Path
import torch
from torch.nn import functional as F
import numpy as np
from sdxl.config import RunConfig
from sdxl.hero_cycle import Tail,read_json,write_json,thumb
from sdxl.run_sdxl_mosaic import _configure_determinism,_decode_latent,_load_pipeline
from sdxl.hero_cycle_render import digest

LATE={5:.17,6:.22,7:.25,8:.22,9:.15}
def lowpass(x,side):
    return F.interpolate(F.adaptive_avg_pool2d(x.float(),(side,side)),size=x.shape[-2:],mode="bilinear",align_corners=False)
def weight(ref):
    luminance=thumb(ref,16) @ [.2126,.7152,.0722]
    value=torch.tensor(np.clip((luminance-.63)/.16,0,1),dtype=torch.float32)[None,None]
    return F.interpolate(F.max_pool2d(value,3,stride=1,padding=1),(128,128),mode="bilinear",align_corners=False)
def encode(pipe,prompt):
    pos,neg,pp,npool=pipe.encode_prompt(prompt=prompt,negative_prompt="",device="cuda",do_classifier_free_guidance=True)
    ids=pipe._get_add_time_ids((1024,1024),(0,0),(1024,1024),dtype=pos.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).cuda()
    return dict(prompt_embeds=pos,negative_prompt_embeds=neg,pooled_prompt_embeds=pp,negative_pooled_prompt_embeds=npool,
                add_time_ids=ids,negative_add_time_ids=ids)
