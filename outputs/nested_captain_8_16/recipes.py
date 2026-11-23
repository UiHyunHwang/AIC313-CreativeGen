from dataclasses import replace
from pathlib import Path
import json
from sdxl.config import RunConfig

root=Path(__file__).resolve().parent
rank=json.loads((root/"selection.json").read_text())
parent=Path(rank["parent"])
cfg=RunConfig.read(parent/"config.json")
winner=rank["ranked"][0]["tile"]
row,col=map(lambda v:int(v[1:]),winner.split("_"))
digest=json.loads((parent/"manifest.json").read_text())["outputs"]["final_latents.safetensors"]["sha256"]
source={"run":str(parent),"sha256":digest,"row":row,"column":col}
for n in (8,16):
    recipe=replace(cfg,prompt=cfg.tile_prompt,tile_prompt="captain america",height=1024*n,width=1024*n,
                   tile_noise_seeds=None,parent_tile_source=source, gpu_memory_limit_gib=None, min_free_gpu_gib=0.0)
    recipe.validate(replay=True)
    case=root/f"inner_{n}"
    case.mkdir(exist_ok=True)
    (case/"recipe.json").write_text(json.dumps(recipe.to_dict(),indent=2)+"\n")
print(winner,source)
