"""Rebuild the original Spider-Man portrait for a clean project checkout."""
import torch
from safetensors.torch import save_file
from outputs.outward_palette16 import common
from outputs.outward_palette16.targets import ROOT,SPIDER,config

PROMPT="A centered front-facing portrait of Spider-Man, his red webbed mask and two large white eyes filling the frame against a charcoal background."
@torch.inference_mode()
def main():
    seed=42
    common._configure_determinism(True)
    old=common.RunConfig.read(ROOT/"recipe.json")
    cfg=common.replace(old,gpu_memory_limit_gib=None,min_free_gpu_gib=0.)
    pipe=common._load_pipeline(cfg,True)
    z=pipe(prompt=PROMPT,negative_prompt="",width=1024,height=1024,num_inference_steps=20,
           guidance_scale=5.,eta=0.,generator=torch.Generator().manual_seed(seed),output_type="latent").images
    image=common._decode_latent(pipe,z,"cpu-tiles")
    SPIDER.parent.mkdir(exist_ok=True)
    image.save(SPIDER)
    save_file({"latents":z.cpu().contiguous()},str(SPIDER.with_suffix(".safetensors")))
if __name__=="__main__":main()
