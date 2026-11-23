import json
from pathlib import Path
from sdxl.config import RunConfig
from dataclasses import replace

def child_recipe(parent, tile, prompt):
    cfg=RunConfig.read(parent/'config.json')
    manifest=json.loads((parent/'manifest.json').read_text())
    assert manifest['status']=='complete'
    r,c=(int(v[1:]) for v in tile.split('_'))
    source=dict(run=str(parent.resolve().relative_to(Path.cwd())),sha256=manifest['outputs']['final_latents.safetensors']['sha256'],row=r,column=c)
    result=replace(cfg,prompt=cfg.tile_prompt,tile_prompt=prompt,parent_tile_source=source,
                   tile_noise_seeds=None,gpu_memory_limit_gib=None,min_free_gpu_gib=0.)
    result.validate(replay=True)
    return result.to_dict()
