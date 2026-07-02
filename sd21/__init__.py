"""Stable Diffusion 2.1 photomosaic with bootstrap, latent upscale, and tiled denoising."""

from sd21.tiling import tiled_tail_denoise

__all__ = ["tiled_tail_denoise"]
