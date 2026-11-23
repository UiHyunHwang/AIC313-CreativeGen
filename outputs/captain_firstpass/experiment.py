"""Paired, single-trajectory SDXL experiment on the outward Spider-Man board."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "outward_hero_cycle"
SOURCE = ROOT.parent / "spider_iron_captain"
sys.path.insert(0, str(SOURCE))
from generate import (DisplayJudge, GridSampler, LUMA, RunConfig, Tail, _configure_determinism,
                      _decode_latent, _load_pipeline, digest, encode, parent, predict,
                      read_json, thumb, write_json)

# Fixed before generation. Positive CLIP margins favor a figure over a shield alone,
# empty scenery or a tiny speck. Luma and RGB errors are measured on [0, 1].
GATE = dict(min_gain88=.005, min_win_fraction=.5, max_native_loss=.005,
            max_luma_increase=.015, max_rgb_increase=.015)
LATE = {5:.17, 6:.22, 7:.25, 8:.22, 9:.15}
BOX = Image.Resampling.BOX
LANCZOS = Image.Resampling.LANCZOS

def metrics(judge, image, ref):
    display = image.resize((128, 128), BOX).resize((88, 88), LANCZOS)
    tiny = image.resize((64, 64), BOX).resize((44, 44), LANCZOS)
    features = judge.model.get_image_features(**judge.proc(images=[display, image, tiny], return_tensors="pt"))
    sims = F.normalize(features) @ judge.text.T
    m = (sims[:, :judge.count].amax(1) - sims[:, judge.count:].amax(1)).tolist()
    delta = thumb(image) - thumb(ref)
    return dict(margin88=m[0], native=m[1], margin44=m[2],
                luma=float(np.abs(delta @ LUMA).mean()), rgb=float(np.abs(delta).mean()))

def lowpass(x, side):
    small = F.adaptive_avg_pool2d(x.float(), (side, side))
    return F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False)

def sample(pipe, tail, crop, seed, condition):
    clean = crop.contiguous().cuda()
    noise = torch.randn(clean.shape, dtype=clean.dtype, generator=torch.Generator().manual_seed(seed)).cuda()
    # Keep the surrounding color and a rough light/dark split while allowing new contours.
    base = lowpass(clean, 2).to(clean.dtype)
    z = pipe.scheduler.add_noise(base, noise, tail.times[:1])
    for k, t in enumerate(tail.times):
        u = predict(pipe.unet, z, t, condition, True)
        direction = predict(pipe.unet, z, t, condition, False) - u
        z_next, x0 = tail.step(u + (6.5 if k < 5 else 5.) * direction, t, z)
        if k in LATE:
            # Bias the clean component of the next noisy state toward the parent's
            # low frequencies. The subsequent UNet steps still generate all detail.
            b = float(tail.alphas[int(t)][1])
            shift = lowpass(clean.float() - x0.float(), 4)
            z_next = (z_next.float() + b**.5 * LATE[k] * shift).to(z.dtype)
        z = z_next
    return _decode_latent(pipe, z)

@torch.inference_mode()
def main():
    stage = read_json(OLD / "spider_man/stage.json")
    edge = read_json(OLD / "report.json")["edges"][0]
    portal = 8*(edge["row"]-1)+edge["column"]-1
    cfg = replace(RunConfig.read(SOURCE/"recipe.json"), gpu_memory_limit_gib=None, min_free_gpu_gib=0.)
    version = digest(Path(__file__))
    write_json(ROOT/"protocol.json", dict(generator_sha256=version, board=stage["subject"], leaves=stage["leaves"],
        source_sha256=stage["source_png"], source_latent_sha256=stage["source_latent"],
        grid=8, native_side=1024, steps=12, start=836, cfg_early=6.5, cfg_late=5.,
        initial_pool=2, late_pool=4, late_weights_zero_based=LATE, gate=GATE,
        excluded_portal=portal, sdxl_model=cfg.model_id, sdxl_revision=cfg.revision))
    _configure_determinism(True)
    pipe = _load_pipeline(cfg, True)
    conditions = {p:encode(pipe,p) for p in stage["prompts"]}
    pipe.maybe_free_model_hooks()
    pipe.remove_all_hooks()
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.unet.to("cuda")
    pipe.vae.to("cuda")
    tail = Tail(pipe.scheduler, 12)
    judge = DisplayJudge(dict(positive=stage["positive"], negative=stage["negative"]))
    target = SOURCE / stage["source"]
    assert digest(target) == stage["source_png"] and digest(target.with_suffix(".safetensors")) == stage["source_latent"]
    canvas, target_image = parent(target.with_suffix(".safetensors"), target)
    for tile in stage["tiles"]:
        i = tile["index"]
        old = OLD / tile["file"]
        assert digest(old) == tile["sha256"]
        out = ROOT / tile["file"]
        out.parent.mkdir(exist_ok=True)
        meta = out.with_suffix(".json")
        if out.exists() and meta.exists():
            record = read_json(meta)
            if record["generator"] == version and digest(out) == record["sha256"] and record["baseline_sha256"] == tile["sha256"]:
                continue
        y, x = (i//8)*128, (i%8)*128
        ref = target_image.crop((x,y,x+128,y+128))
        image = sample(pipe,tail,canvas[...,y:y+128,x:x+128],tile["seed"],conditions[tile["prompt"]])
        image.save(out,compress_level=2)
        with Image.open(old) as original:
            baseline = metrics(judge,original.convert("RGB"),ref)
        write_json(meta,dict(index=i, file=tile["file"], seed=tile["seed"], prompt=tile["prompt"],
            generator=version, sha256=digest(out), baseline_sha256=tile["sha256"],
            baseline=baseline, candidate=metrics(judge,image,ref)))
        if (i+1)%4 == 0: print(f"{i+1}/64",flush=True)
    print("Peak reserved GiB", torch.cuda.max_memory_reserved()/2**30,flush=True)

if __name__ == "__main__":
    main()
