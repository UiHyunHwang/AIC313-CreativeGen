# Captain America tiles in the outward Spider-Man mosaic

This experiment compares two SDXL generation schedules at matching coordinates, prompts and seeds on the outward Spider-Man board. The target is a previously generated Spider-Man portrait. Its PNG and latent hashes are checked against the original generation record. The two other boards and the camera settings come from the outward cycle.

The candidate uses a 2×2 low-pass version of the original target latent crop to initialize the tile. The first five updates use classifier-free guidance 6.5. The remaining seven use 5.0. During updates 6 through 10, a 4×4 low-frequency residual from the target latent biases the next noisy state. Two final UNet updates finish the image. Each of the 64 coordinates gets one completed 1024×1024 generation. Seeds range from 8721000 through 8721063. The three existing prompts stay tied to their original coordinates. Every change is applied inside the diffusion trajectory. CLIP is only used to evaluate completed candidates in this variant.

The paired evaluation uses the completion samples from the old schedule as controls. At the GIF's 88-pixel cell size, the positive CLIP descriptions identify Captain America, and the negative descriptions describe an isolated shield or flag, empty scenery or a tiny figure. The CLIP margin is the best positive similarity minus the best negative similarity. The native and 44-pixel scores use the same text. The tone and RGB errors measure deviations from the target on a 4×4 grid per tile, equivalent to a 32×32 representation of the whole board before portal insertion. Every position has equal weight. Thresholds in protocol.json were set before generation. comparison.json reports all non-portal positions, a breakdown by prompt, and a sensitivity check on all 64 cells. Bootstrap intervals resample the coordinates and describe this board only. The paired gallery is in row-major order and includes the uncomposited portal (marked with an asterisk).

Run from the PhotoQuilt root after reproducing the baseline as described in outputs/outward_hero_cycle/README.md (also in its reproduction.zip). Use the saved local SDXL and CLIP model revisions and the creativegen environment.

```bash
export PYTHONPATH=. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python outputs/captain_firstpass/experiment.py
python outputs/captain_firstpass/review.py --render
```

The second command writes the paired galleries and summary. A passing candidate also goes through the original automatic portal-selection and rendering procedure, with another check on the actual visible tile set. All outputs are written inside this experiment directory.

The generator is Podell et al., SDXL, checkpoint stabilityai/stable-diffusion-xl-base-1.0 (revision in protocol.json). The verifier is Radford et al., CLIP, checkpoint openai/clip-vit-base-patch32, revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. The project uses the PhotoQuilt paper and code, Diffusers and the earlier project modules cited in the baseline reproduction archive. Code and earlier prompt brainstorming used Codex. This experiment uses prompt and seed inputs with the team's SDXL-generated target and does not use an external reference dataset or train model weights.
