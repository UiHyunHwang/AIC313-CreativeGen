# Reserve a recursive zoom cell during SDXL sampling

The outward order is Spider-Man → Iron Man → Captain America → Hulk → Thor → Spider-Man. Edit the ten short prompts and the new master seed in [reserved_cycle.json](reserved_cycle.json). Every board uses the preceding character's tile prompt across all 256 cells. The target seeds in camera order are 92480, 93481, 93482, 92483 and 93484. The tile-noise tensor uses seed 92485. The only visual inputs are produced inside this run.

## Reproduce

Use the versions in [hero_cycle_requirements.txt](hero_cycle_requirements.txt) and the pinned official SDXL base cache documented in [JOINT_ZOOM.md](JOINT_ZOOM.md). Cache the feature encoder once via:

```bash
huggingface-cli download facebook/dinov2-small --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 --include config.json model.safetensors preprocessor_config.json
```

From the repository root, with the creativegen environment active:

```bash
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m sdxl.reserved_cycle outputs/reserved_iron_tiles16
python -m sdxl.render_reserved_cycle outputs/reserved_iron_tiles16 --mode shared-latent
python -m sdxl.verify_joint_cycle outputs/reserved_iron_tiles16/shared-latent
python -m unittest discover -s tests
```

The sampler checkpoints all five boards after each timestep. On restart it reloads only this recipe's targets and the most recent completed step. The output contains the resolved prompts, model revisions, seeds, reservation costs, positions, and memory and camera records.

The `target_guidance` field belongs to the outer portrait in each recipe row. Starting at step 4, the sampler pools each tile's target-latent residual to a square grid. It upsamples the correction and applies it through the inclusive `last_step`. The Captain board uses gain 0.28 and an 8×8 correction grid through step 8, leaving four SDXL updates for tile details. The other portraits use gain 0.23 and a 5×5 grid through step 10. Each board contains 16×16 tiles.

For the Iron Man tiles, five pilot settings each generated the same 32 ordinary cells with the production noise seed, for 160 pilot tile samples. Two cells per row covered the full board height. The comparison used the previous production tiles at the same coordinates, magnified tile contact sheets, CLIP margins against four other heroes and coarse target error. The short workshop film-still prompt with guidance through step 8 gave distinct figures in both medium and wide shots. The production cycle generates every tile from its prompt and seed. Generation and selection details for the pilot are recorded in `iron_tile_validation.json` in the local output.

The previous and current Captain boards share 254 ordinary cell coordinates because their reserved cells differ. Across these cells the tenth percentile of the Iron Man CLIP margin rose from 0.0260 to 0.0367, while the overview's luminance-gradient similarity to the target at 128px fell from 0.906 to 0.793. When Iron Man gives way to Captain America, the camera pulls back from row 14, column 12 of the Captain board.

## How a cell is chosen

1. At step 4 of 12, use the official SDXL base VAE to decode inexpensive 256px previews from reduced clean latent predictions. Assemble a small preview of each inner board. These previews precede the final ordinary tile images.
2. Feed the previews to frozen DINOv2-S/14 at 224px. For each of the 256 candidates score spatial patch-token distance (50%), global image-token distance (15%), coarse RGB difference (20%) and color mismatch with the facing edges of the neighboring cells (15%). Select the minimum once. The full scores and tie order (row-major) are recorded. DINOv2 supplies features only.
3. Starting with the next update, SDXL denoises the 255 ordinary tiles of each board. The reserved cell receives a pooled latent view of the previous board during generation. The five references are resolved simultaneously; each additional nesting shrinks by a factor of 16. Facing edges of neighboring tiles receive a tapered latent correction through step 11. This is one shared generation with 1,275 final ordinary leaves, five reserved cells and five generated targets. All leaves and targets are retained.
4. In shared-latent output, the official base VAE encodes and decodes the current inner image to generate the reserved cell at final rasterization. Four simultaneous passes account for the cycle. The encoder uses the deterministic posterior mode. The resulting cut error is measured in render.json. The GIF is made of crops and resizes with one fixed palette. It holds for 5.5 seconds per subject and moves for 4.5 seconds per leg.

The exporter also implements `--mode shared-rgb`. In this variant an empty reserved cell displays the actual reduced RGB view of the inner board and their pixels coincide exactly. The course's rules limit post-processing; confirm this particular use of exact image references with the teaching staff before using this variant for submission. The shared-latent and shared-rgb variants have distinct output folders and separate provenance records.

References include the [DINOv2 paper](https://arxiv.org/abs/2304.07193), [Meta repository](https://github.com/facebookresearch/dinov2), [pinned feature encoder](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056) and [the project's SDXL, DDIM, PhotoQuilt and software references](JOINT_ZOOM.md#references). Disclose AI assistance for coding and prompt brainstorming in the write-up. CLIP is used only for final diagnostics. Selection scores, seed counts, code, and citations belong with any submitted output. See the [course instructions](https://generative-models.kaist.ac.kr/creativegen/).

## Earlier Captain silhouette run

The Captain setting came from a paired comparison of eight parameter combinations across 68 tiles in the face, chest and shield regions. In the full render, the cosine similarity between the target and board luminance gradients at a 128px viewing size increased from 0.686 to 0.915 in the face, 0.471 to 0.851 in the chest and 0.529 to 0.817 in the shield. All five DINOv2 reservations matched the earlier run. The Captain-to-Hulk transition's 128px RGB MAE increased from 0.0223 to 0.0247. The verified 50-second GIF has 451 frames and SHA-256 `a8941861590a8d3ee9370b9804f20d759a67bc7eefc2fae985769fb149868b7f`.

## Earlier first-draw target refresh (commit `b3ac475`)

This run used one draw per character, with all seeds fixed before generation. Spider-Man and Hulk used their previous seeds. Their regenerated PNGs and target latents matched the previous run byte for byte. For Iron Man, Captain America and Thor, the target seeds increased by 1000 for each character. The target and tile prompts, the tile noise seed and the sampling settings matched commit `1eab647`.

A separate exploratory run generated 18 previews before the first-draw protocol was requested. Its six seed rounds used `92480 + 100*k + j`, with k from 1 through 6 and zero-based hero indices j = 1, 2, 4. None supplied a target, score or seed for this production run. The five production targets came from five independent prompt-and-seed calls.

The verified shared-latent GIF contains 451 frames over 50 seconds and passes the loop-boundary check. Its SHA-256 is `eb0c7963b5c25d77e1d5318fe52bd5d8c70d666ad7066fade88ab02e6490a4a9`. The intended tile subject ranked first for 1246/1280 cells on the five-label CLIP evaluation.

## Earlier close portraits with a single location (commit `1eab647`)

Compared with commit `7ad3c5d`, only the five target prefixes change from `Portrait of` to `Close portrait of`. The run used the first target from each of the five recorded seeds. The decoded GIF has 451 frames, runs for 50 seconds and passes the loop-boundary check. CLIP ranked the intended tile subject first in 1247/1280 cells among the five labels. The GIF SHA-256 is `24fcc488d0bbc0961e1059bf0751a09012401121cfb47b6dbd684978ab67e56d`.

## Earlier single-location run (commit `7ad3c5d`)

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
