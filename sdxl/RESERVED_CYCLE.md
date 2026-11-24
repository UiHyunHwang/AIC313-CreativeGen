# Reserve a recursive zoom cell during SDXL sampling

The outward order is Spider-Man → Iron Man → Captain America → Hulk → Thor → Spider-Man. Edit the ten short prompts and the new master seed in [reserved_cycle.json](reserved_cycle.json). Every board uses the preceding character's tile prompt across all 256 cells. The five targets are generated from seeds 92480–92484 and the tile-noise tensor from 92485. The only visual inputs are produced inside this run.

## Reproduce

Use the versions in [hero_cycle_requirements.txt](hero_cycle_requirements.txt) and the pinned official SDXL base cache documented in [JOINT_ZOOM.md](JOINT_ZOOM.md). Cache the feature encoder once via:

```bash
huggingface-cli download facebook/dinov2-small --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 --include config.json model.safetensors preprocessor_config.json
```

From the repository root, with the creativegen environment active:

```bash
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m sdxl.reserved_cycle outputs/reserved_locations16
python -m sdxl.render_reserved_cycle outputs/reserved_locations16 --mode shared-latent
python -m sdxl.verify_joint_cycle outputs/reserved_locations16/shared-latent
python -m unittest discover -s tests
```

The sampler checkpoints all five boards after each timestep. On restart it reloads only this recipe's targets and the most recent completed step. The output contains the resolved prompts, model revisions, seeds, reservation costs, positions, and memory and camera records.

## How a cell is chosen

1. At step 4 of 12, use the official SDXL base VAE to decode inexpensive 256px previews from reduced clean latent predictions. Assemble a small preview of each inner board. These previews precede the final ordinary tile images.
2. Feed the previews to frozen DINOv2-S/14 at 224px. For each of the 256 candidates score spatial patch-token distance (50%), global image-token distance (15%), coarse RGB difference (20%) and color mismatch with the facing edges of the neighboring cells (15%). Select the minimum once. The full scores and tie order (row-major) are recorded. DINOv2 supplies features only.
3. Starting with the next update, SDXL denoises the 255 ordinary tiles of each board. The reserved cell receives a pooled latent view of the previous board during generation. The five references are resolved simultaneously; each additional nesting shrinks by a factor of 16. Facing edges of neighboring tiles receive a tapered latent correction through step 11. This is one shared generation with 1,275 final ordinary leaves, five reserved cells and five generated targets. All leaves and targets are retained.
4. In shared-latent output, the official base VAE encodes and decodes the current inner image to generate the reserved cell at final rasterization. Four simultaneous passes account for the cycle. The encoder uses the deterministic posterior mode. The resulting cut error is measured in render.json. The GIF is made of crops and resizes with one fixed palette. It holds for 5.5 seconds per subject and moves for 4.5 seconds per leg.

The exporter also implements `--mode shared-rgb`. In this variant an empty reserved cell displays the actual reduced RGB view of the inner board and their pixels coincide exactly. The course's rules limit post-processing; confirm this particular use of exact image references with the teaching staff before using this variant for submission. The shared-latent and shared-rgb variants have distinct output folders and separate provenance records.

References include the [DINOv2 paper](https://arxiv.org/abs/2304.07193), [Meta repository](https://github.com/facebookresearch/dinov2), [pinned feature encoder](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056) and [the project's SDXL, DDIM, PhotoQuilt and software references](JOINT_ZOOM.md#references). Disclose AI assistance for coding and prompt brainstorming in the write-up. CLIP is used only for final diagnostics. Selection scores, seed counts, code, and citations belong with any submitted output. See the [course instructions](https://generative-models.kaist.ac.kr/creativegen/).

## Single-location target prompts

The five target prompts each name a character and one place. The tile prompts, seeds, model, sampler, reservation rule and camera settings match commit `454858e`. All five first-draw targets were used. Each new board has 16×16 cells. CLIP ranked the intended tile subject first in 1250/1280 displayed cells on the five-label comparison. The exported GIF contains 451 frames over 50 seconds and passed the loop-boundary check. Its SHA-256 is `91a6f8c20385f62a95cd8de278da3726f930726be35fbced2073cbd1218383ea`.

## Earlier portrait run (commit `454858e`)

These figures describe the prompts saved with `outputs/reserved_outward16`.

The shared-latent export has 451 frames at 704×704, lasts 50 seconds, and passed the decoded GIF loop-boundary check. Five targets and all five board overviews ranked first for their intended subject against the five CLIP labels. Of 1,280 displayed cells, 1,232 ranked first for their intended tile subject. Peak PyTorch reserved memory during sampling was 12.12 GiB on the GRID A100D-20C. The course staff had not yet confirmed the exact RGB variant, which was tested only with synthetic unit-test rasters.

| Reveal | Reserved row and column (1-based) | Portal-to-child 128px RGB MAE |
| --- | --- | ---: |
| spider_man → iron_man | 11, 2 | 0.0214 |
| iron_man → captain_america | 2, 11 | 0.0220 |
| captain_america → hulk | 6, 13 | 0.0255 |
| hulk → thor | 16, 12 | 0.0175 |
| thor → spider_man | 15, 15 | 0.0225 |

The GIF SHA-256 is `25af82592cbe9bfc3dd731189a916cbb6ce1a4a366f347afdf7a6b3d0ae116e3`.
