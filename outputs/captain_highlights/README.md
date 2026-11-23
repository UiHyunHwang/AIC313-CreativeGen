# Bright contour preservation in the outward cycle

This follow-up was designed after inspecting the subject-first board. At 88 pixels Captain America became easier to identify, but some of Spider-Man's bright eye outlines became muted. A single extra generation rule uses the luminance of the SDXL-generated target before any tile is sampled. A 16×16 area average per tile, smooth intensity ramp from 0.63 to 0.79, one-cell expansion and bilinear interpolation yield a continuous weight. This weight preserves more of the parent latent at initialization and adds some finer parent structure during five middle-to-late diffusion updates. All positions use the same formula. When its weight is identically zero the two schedules are identical and the program reuses the first experiment's exact PNG. The renderer receives the complete board from one schedule.

The original paired character, color and tone checks still apply. The additional tests use a 64×64 luminance representation of the board, comparing all cells as well as the subset brighter than 0.70 in the original SDXL target. Recover at least half the average highlight brightness lost in the first experiment and keep mean absolute luminance error within 0.002 of the original board. These thresholds were specified before running this follow-up. The script records the diagnostics both before and, on successful render, after inserting the recursive child at its automatically chosen position.

From the PhotoQuilt project root first follow the prerequisites and both commands in outputs/captain_firstpass/README.md. Then run in the same creativegen environment:

```bash
export PYTHONPATH=. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python outputs/captain_highlights/generate_highlights.py
python outputs/captain_highlights/review_highlights.py
```

The parent directory structure in the reproducibility bundle matches the project. Extract outward_hero_cycle/reproduction.zip in the project as described in the included baseline README, then extract its two prerequisite archives at the project root. Tile JSONs record the prompt, seed, paired scores and PNG hashes. The licensed model weights should be supplied through the pinned Hugging Face cache in the existing environment. No training or external visual input was used in either experiment. Code and earlier prompt brainstorming used Codex. See the first experiment and baseline READMEs for provenance and model and paper credits.

The paired study generated 64 full first-pass tiles and 23 additional trajectories with nonzero contour weight. The final board uses the contour-preserving schedule on all 64 coordinates (41 equivalent cached results). Peak PyTorch GPU reserved memory was 10.94 GiB in each run. The render was validated at 268 frames and 30.02 seconds, with 5.5-second stops, infinite looping and pixel-identical decoded first and last frames.
