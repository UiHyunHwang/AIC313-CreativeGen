# Three outward reveals

The GIF opens with Spider-Man and pulls back to an Iron Man built from Spider-Men. The camera pulls farther back to a Captain America built from Iron Men, then farther back to the opening Spider-Man built from Captain Americas. Each moving leg expands the camera field from one cell to the entire 8×8 parent. The endpoint at each leg matches the start of the next one. The gallery zoom_sequence.jpg shows the start, middle and end of each move.

The Iron Man board uses the 64 generated SDXL leaves from dualscale_reveal. Two new boards use the previously generated Spider-Man and Captain America portraits as targets, with 64 new native SDXL generations each. Seeds, all prompts, checkpoint details in the inherited recipes, and generation-time branch choices can be found in the accompanying records. Each parent receives its actual child overview in one fixed grid cell selected by color and boundary error. The script resolves these cyclic thumbnails until pixel-stable and checks the decoded GIF stops and wrap. The camera operates on these linked images with classical resampling.

From the PhotoQuilt project root, with both prerequisites already reproduced or present, run:
```bash
export PYTHONPATH=. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python outputs/outward_hero_cycle/generate_outward.py
python outputs/outward_hero_cycle/render_outward.py
```

For a clean rebuild, extract the two prerequisite ZIPs into the same clean project root. Use the previously pinned creativegen package versions and cached official SDXL base and non-generative CLIP revisions. Follow outputs/dualscale_pilot/README.md to generate the original Iron Man target. Run outputs/dualscale_reveal/generate.py for the Iron Man board, then outputs/spider_iron_captain/generate.py for the other two target inputs. Follow the two commands above for the final result. The optional historical outward_preview.gif uses a separately existing hero_cycle_varied animation; it is not used to make the final 8×8 GIF. Code and text prompt brainstorming used Codex. Both prerequisite archives contain the model, code and paper citations.
