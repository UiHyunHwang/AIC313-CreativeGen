import json
from pathlib import Path
from dataclasses import replace
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from sdxl.config import RunConfig
from sdxl.run_sdxl_mosaic import _configure_determinism, _load_pipeline, _decode_latent
from sdxl.tiling import tiled_tail_denoise
from sdxl.memory_guard import GpuMemoryGuard

out = Path(__file__).resolve().parent
cfg = RunConfig.read(out/"config.json")
coords = [(0,0),(1,4),(4,3),(7,4)]
cases = [(20,0),(13,0),(12,.25),(12,.60)]
sheet = Image.new("RGB",(1250,1030),"white")
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
for i,title in enumerate(["Target crop","Original  t=550","Higher noise  t=836","2x2 palette  0.25","2x2 palette  0.60"]):
    draw.text((i*250+6,6),title,font=font,fill="black")
with GpuMemoryGuard(3,out), torch.inference_mode():
    _configure_determinism(True)
    pipe = _load_pipeline(replace(cfg,offload="model"),True)
    pos,neg,pool,n_pool = pipe.encode_prompt(prompt=cfg.tile_prompt,negative_prompt="",device="cuda",do_classifier_free_guidance=True)
    tids=pipe._get_add_time_ids((1024,1024),(0,0),(1024,1024),dtype=pos.dtype,text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to("cuda")
    pipe.maybe_free_model_hooks()
    z = load_file(out/"target_latents.safetensors")["latents"].cuda()
    up = torch.nn.functional.interpolate(z.float(),size=(1024,1024),mode="bicubic",align_corners=False).half()
    eps = torch.randn(up.shape,generator=torch.Generator().manual_seed(43),dtype=up.dtype).cuda()
    with Image.open(out/"target.png") as target:
        for j,(r,c) in enumerate(coords):
            crop=target.crop((128*c,128*r,128*(c+1),128*(r+1)))
            sheet.paste(crop.resize((242,242),Image.Resampling.BICUBIC),(6,38+j*248))
    for i,(steps,gain) in enumerate(cases,1):
        pipe.scheduler.set_timesteps(steps,device="cuda")
        ts=pipe.scheduler.timesteps[-12:]
        for j,(r,c) in enumerate(coords):
            sl=(...,slice(r*128,(r+1)*128),slice(c*128,(c+1)*128))
            clean,noise=up[sl].contiguous(),eps[sl].contiguous()
            initial=noise if gain else pipe.scheduler.add_noise(clean,noise,ts[:1])
            final=tiled_tail_denoise(latents=initial,timesteps=ts,scheduler=pipe.scheduler,unet=pipe.unet,
                prompt_embeds=pos,negative_prompt_embeds=neg,pooled_prompt_embeds=pool,negative_pooled_prompt_embeds=n_pool,
                add_time_ids=tids,negative_add_time_ids=tids,tile_batch=1,tile_latent=128,serial_cfg=True,
                guidance_scale=5,progress=False,palette_target=clean if gain else None,palette_strength=gain)
            pipe.maybe_free_model_hooks()
            image=_decode_latent(pipe,final)
            sheet.paste(image.resize((242,242),Image.Resampling.LANCZOS),(6+i*250,38+j*248))
        sheet.save(out/"comparison.jpg",quality=93)
        print(f"Completed schedule={steps} gain={gain}",flush=True)
