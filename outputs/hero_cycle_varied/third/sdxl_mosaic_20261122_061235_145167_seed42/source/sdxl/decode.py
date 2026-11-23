"""VAE tile decoding with CPU accumulation, preserving Diffusers' blending.

Only decoder computation uses the GPU. Previously decoded tiles and the final
pixel canvas stay in host memory, so GPU pixel storage does not grow with canvas
area. This uses the official SDXL VAE, not additional generative weights.
"""

from __future__ import annotations

import torch


@torch.inference_mode()
def decode_tiles_to_cpu(vae, latents: torch.Tensor) -> torch.Tensor:
    tile_size = int(vae.tile_latent_min_size)
    stride = int(tile_size * (1 - vae.tile_overlap_factor))
    blend = int(vae.tile_sample_min_size * vae.tile_overlap_factor)
    keep = int(vae.tile_sample_min_size) - blend
    scale = int(vae.tile_sample_min_size) // tile_size
    if stride <= 0 or keep != stride * scale:
        raise ValueError("Unsupported VAE tiling geometry")
    if latents.shape[0] != 1:
        raise ValueError("CPU tile decoding currently supports one image")
    height, width = latents.shape[-2:]
    output = None
    previous_row = []
    row_positions = list(range(0, height, stride))
    for row_index, top in enumerate(row_positions):
        current_row = []
        for column, left in enumerate(range(0, width, stride)):
            block = latents[:, :, top:top + tile_size, left:left + tile_size]
            # Block size never exceeds the native VAE tile limit, so decode()
            # runs one block and still activates Accelerate's offload hooks.
            decoded_gpu = vae.decode(block, return_dict=False)[0]
            tile = decoded_gpu.to(device="cpu", dtype=torch.float32)
            del decoded_gpu
            if not torch.isfinite(tile).all().item():
                raise RuntimeError("VAE decoding produced non-finite pixels")
            if row_index:
                tile = vae.blend_v(previous_row[column], tile, blend)
            if column:
                tile = vae.blend_h(current_row[column - 1], tile, blend)
            current_row.append(tile)
            if output is None:
                output = torch.empty((1, tile.shape[1], height * scale, width * scale), dtype=tile.dtype)
            patch = tile[:, :, :keep, :keep]
            y, x = top * scale, left * scale
            output[:, :, y:y + patch.shape[2], x:x + patch.shape[3]] = patch
        previous_row = current_row
        print(f"  VAE decode row {row_index + 1}/{len(row_positions)} (CPU accumulation)", flush=True)
    return output
