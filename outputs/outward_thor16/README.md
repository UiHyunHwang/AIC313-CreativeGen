# Thor in the four-stage outward cycle

The film pulls out from Spider-Man to Iron Man, Captain America, Thor and the opening Spider-Man. Each native parent measures 16,384 × 16,384 pixels. Its 16×16 leaves are native 1024-pixel SDXL outputs. The inherited camera holds each hero for 5.5 seconds, then pulls back for 4.5 seconds. The encoded last frame equals the encoded first frame. The GIF uses the approved three-stage film's color palette.

Iron Man's Spider-Man tiles and Captain America's Iron Man tiles are reused from the approved outward_scene16 run, with their original two portal coordinates. Their linked previews are resolved again for the four-stage cycle. The renderer compares every preserved ordinary cell against its stored PNG hash and compares the two completed lossless overviews against the original overviews outside the linked cells.

The Spider-Man portrait and its target latent come from the previous run. Its new leaves are Thor. The new Thor portrait is composed of Captain America. The code generates four Thor portrait candidates from the same prompt, then selects the smallest combined color cost for the Captain America to Thor and Thor to Spider-Man links plus the Captain America tile palette. All four seeds, scores and hashes are recorded. The tile sampler uses the same local color, scene conditioning, 12-step schedule and bright-edge preservation as the approved Spider-Man board. Each affected coordinate uses a single trajectory, for 512 new native tile samples. The automatically selected portal occupies one cell per finished board. The render and verification scripts check the cycle, resolution, content hashes, original cell positions, stop timings and GIF loop. CLIP assesses identity at display size and does not select finished tiles.

## Reproduction

Extract reproduction.zip at the project root. Use the pinned creativegen environment and model cache. Follow the included outputs/outward_scene16/README.md to reproduce the approved three-stage run if it is not already present. Then run:

```bash
export PYTHONPATH=. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python outputs/outward_thor16/portraits.py
python outputs/outward_thor16/new_boards.py
python outputs/outward_thor16/render_cycle.py
python outputs/outward_thor16/verify.py
```

The code generates all target and tile images using the official SDXL base. The bundle contains the three-stage source and audit records; generating the four-stage GIF needs the prior raw tile outputs or their reproduction. The source archive can itself be recreated by verify.py when the prior run's archive is present. Code and prompt brainstorming used Codex. The previous run README includes credits to PhotoQuilt, SDXL, Diffusers, PyTorch and CLIP.
