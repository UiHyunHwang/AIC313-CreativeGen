"""Fixed-tile (MultiDiffusion-style) denoising for the SD 2.1 mosaic tail.

The flux1/flux2 mosaic denoises the tail in fixed-size attention windows and
merges overlapping windows with a scatter-mean. SD 2.1 is a UNet model, so the
faithful analog is to run the UNet on each fixed-size latent window every tail
step and average the per-window epsilon predictions back onto a single global
latent (one shared scheduler trajectory keeps the canvas globally coherent while
each window denoises like a native-resolution generation).
"""

from __future__ import annotations

from typing import List

import torch


def window_starts(total: int, win: int, stride: int) -> List[int]:
    """Start indices covering ``total`` with ``win``-sized windows, edge-flushed.

    The final window is clamped to end exactly at the edge, which (on canvases
    that don't divide evenly, or with overlap) creates overlap handled by the
    scatter-mean merge.
    """
    if win >= total:
        return [0]
    stride = max(1, stride)
    starts = list(range(0, total - win + 1, stride))
    if not starts or starts[-1] != total - win:
        starts.append(total - win)
    return starts


@torch.no_grad()
def tiled_tail_denoise(
    *,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    inject_index: int,
    unet,
    scheduler,
    cond_embeds: torch.Tensor,
    uncond_embeds: torch.Tensor,
    guidance_scale: float,
    tile_latent: int,
    stride_latent: int,
    tile_batch: int,
    progress: bool = True,
) -> torch.Tensor:
    """Denoise ``latents`` from ``timesteps[inject_index]`` to the end in tiles.

    ``latents`` is the full noisy canvas latent ``(1, C, H, W)``. Returns the
    fully denoised canvas latent.
    """
    _, _, lh, lw = latents.shape
    ys = window_starts(lh, tile_latent, stride_latent)
    xs = window_starts(lw, tile_latent, stride_latent)
    coords = [(y, x) for y in ys for x in xs]
    n_windows = len(coords)

    do_cfg = guidance_scale > 1.0
    tail = list(range(inject_index, len(timesteps)))
    for n, i in enumerate(tail):
        t = timesteps[i]
        eps_acc = torch.zeros_like(latents)
        counts = torch.zeros((1, 1, lh, lw), device=latents.device, dtype=latents.dtype)

        for start in range(0, n_windows, tile_batch):
            batch_coords = coords[start : start + tile_batch]
            wins = torch.cat(
                [latents[:, :, y : y + tile_latent, x : x + tile_latent] for (y, x) in batch_coords],
                dim=0,
            )
            model_in = torch.cat([wins, wins], dim=0) if do_cfg else wins
            model_in = scheduler.scale_model_input(model_in, t)

            bsz = wins.shape[0]
            if do_cfg:
                embeds = torch.cat(
                    [uncond_embeds.expand(bsz, -1, -1), cond_embeds.expand(bsz, -1, -1)], dim=0
                )
            else:
                embeds = cond_embeds.expand(bsz, -1, -1)

            noise_pred = unet(model_in, t, encoder_hidden_states=embeds).sample
            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            for j, (y, x) in enumerate(batch_coords):
                eps_acc[:, :, y : y + tile_latent, x : x + tile_latent] += noise_pred[j : j + 1]
                counts[:, :, y : y + tile_latent, x : x + tile_latent] += 1.0

        eps = eps_acc / counts.clamp_min(1.0)
        latents = scheduler.step(eps, t, latents).prev_sample
        if progress:
            print(f"  tail step {n + 1}/{len(tail)} (t={int(t.item())}, {n_windows} tiles)", flush=True)

    return latents
