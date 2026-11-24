# Outward five-hero joint diffusion cycle

Generate five new SDXL targets from text. In outward playback the order is Spider-Man, Iron Man, Captain America, Hulk, Thor, Spider-Man. The first board has Thor tiles, the second has Spider-Man tiles, and the sequence wraps around. Every board has 16 rows and 16 columns. Each of the 1,280 tiles has an independent slice of the seeded Gaussian noise tensor. A tile is sampled at 768×768; the boards are 12,288×12,288.

## Reproduction

Use the offline model cache and environment in [JOINT_ZOOM.md](JOINT_ZOOM.md). From the repository root:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
python -m sdxl.joint_cycle outputs/joint_outward16 --arm detail
python -m sdxl.render_joint_cycle outputs/joint_outward16
python -m sdxl.verify_joint_cycle outputs/joint_outward16
```

The production command generates its own targets, maps their colors to lighting prompts and records every prompt. It checkpoints all five boards at the end of each DDIM step. The same command resumes an interrupted run. A completed board export contains its native raster, a 1024-pixel overview and the decoded cell connecting it to the previous board.

## Experiment and camera

To reproduce the profile selection, run the 4×4 diffusion-location pilot and its gentle arm from [JOINT_ZOOM.md](JOINT_ZOOM.md). Then run:

```bash
for arm in balanced detail; do
    python -m sdxl.joint_zoom --grid 4 --seed 31415 --slot-policy diffusion --arm "$arm" --output outputs/joint_zoom_diffusion
done
python -m sdxl.select_cycle_guidance outputs/joint_zoom_diffusion
```

All five comparisons use the same noise, prompts, targets and location. In this pilot, native tiles are 1024 pixels. Eligibility requires a positive Captain America CLIP margin and maximum board-to-target MAE within 0.015 of the independent arm. Minimize the sum of the 32- and 128-pixel portal-to-child RGB MAEs among eligible profiles.

| Profile | Gain | Last guided step (1-based) | Spatial bands per axis | 32px MAE | 128px MAE | Captain CLIP margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent | 0 | – | – | .20765 | .22323 | .06784 |
| Gentle | .35 | 8 | 8 | .14913 | .16897 | .07408 |
| Balanced | .58 | 10 | 16 | .07262 | .10072 | .06409 |
| Original strong | .78 | 11 | 32 | .01219 | .06028 | .04683 |
| Detail | .70 | 11 | 64 | .01245 | .03927 | .03920 |

Production uses the selected detail profile on every edge. All five boards predict clean latents before any cross-scale update. At step four each outer board reserves the cell with the nearest pooled latent prediction to the preceding board. On steps four through eleven the frozen SDXL sampler receives the same bidirectional consistency correction as the two-board pilot, including from Thor into Spider-Man. Use the selection experiment as a single-seed pilot; it did not compare all five profiles at production resolution.

The movie consists of crops and resizes of the independently decoded views. A 5.5-second hold starts on a centered crop covering 90% of each board. During each 4.5-second move, the camera first backs out to the full inner board, then cuts in motion to the corresponding moving crop of the outer board. The camera always uses one generated view per frame. The first and last display frames coincide. The measured cut errors and the loop, timing and raster checks are in render.json. Quality evaluation classifies all tiles against five CLIP labels and records each 32px overview classification and target error.

## Submission record

The five target seeds are 72460–72464, with one shared tile-noise tensor drawn from seed 72465. All five production boards are retained. The two additional pilot profiles add 32 tiles each to the earlier pilot, with deterministic regeneration of the pilot targets in each invocation. The previous movie supplies its hero sequence and timing only. The versioned text prompts, generated manifests and reproduction steps accompany the artwork.

Disclose Codex use for coding and prompt brainstorming. Generation uses only the official SDXL base checkpoint with its bundled text encoders and VAE. CLIP is used for scoring. See [the pilot references and environment](JOINT_ZOOM.md#references) for PhotoQuilt, Generative Powers of Ten, SDXL, DDIM, CLIP, model checkpoints and open-source dependencies. The course requires the team to provide the poster, write-up, citations and reproducible source for submitted outputs.

## Recorded production results

All five generated targets and five 32-pixel board overviews scored highest for their intended hero among the five labels. The production run used 8.93 GiB peak PyTorch-reserved GPU memory. The encoded 704×704 GIF contains 451 frames, runs for 50 seconds and ends on its initial frame.

| Reveal | Cell (row, column, 1-based) | 32px cut MAE | 128px cut MAE | Intended hero ranked first among outer tiles |
| --- | --- | ---: | ---: | ---: |
| spider_man → iron_man | 2, 9 | 0.0259 | 0.0619 | 254/256 |
| iron_man → captain_america | 7, 6 | 0.0304 | 0.0574 | 256/256 |
| captain_america → hulk | 2, 16 | 0.0199 | 0.0542 | 222/256 |
| hulk → thor | 5, 10 | 0.0223 | 0.0589 | 255/256 |
| thor → spider_man | 16, 1 | 0.0193 | 0.0529 | 251/256 |

GIF SHA-256: `2602ec03f92c2e788b84698379ef3fd726dbbb8139a45c460f2d2e16b00b248e`.
