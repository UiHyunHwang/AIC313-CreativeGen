# Spider-Man to Iron Man reveal

The 8192px square board contains 64 adjoining 1024px SDXL leaves. Each leaf is generated once with the tone-priority method in the six-location dual-scale study, using the corresponding crop of the team's generated Iron Man latent. The crop for an 8×8 board uses the opening latent upsampled to 1024×1024 latent cells. The sampler evaluates its trajectories during denoising at updates 3, 6 and 9. Each of those decisions scores five clean predictions using a 44px CLIP margin and a four-by-four luminance error weighted by 2. Final diffusion results occupy their native grid positions. The camera stays on a single rendered board. It starts at the cell with the best minimum of native and 44px CLIP margins. The two endpoint views hold for 5.5 seconds each. A 5s zoom out reveals the full board, then a 4.5s zoom in closes the loop.

From the PhotoQuilt root, in the creativegen environment with the cached SDXL base and CLIP revisions, run:

```bash
export PYTHONPATH=. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python outputs/dualscale_reveal/generate.py
python outputs/dualscale_reveal/render.py
```

For a fresh reproduction, first regenerate the seed-42 opening latent using the command in outputs/dualscale_pilot/README.md. That document lists model, method and code credits. To also reproduce the six-case comparison GIF, run outputs/dualscale_pilot/run.py followed by outputs/dualscale_pilot/animate_pilot.py. The prompt and implementation were developed with Codex. The report records the GIF checksum, verified timings and camera start. generation.json records each seed and every branch decision.
