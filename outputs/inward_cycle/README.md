# Four forward zooms

The supplied prior.zip contains the SDXL generation pipeline, pinned model revisions, source citations and prior experiment recipes. Rebuild the archived runs from that bundle or reuse their generated outputs. Use the prior quality variant's output directory as --prior. It must contain quality.json, spider_review.json, report.json, and new_board/ with the complete fourth SDXL run. Sibling directories hero_cycle, hero_cycle_varied and captain_refined follow the names in its instructions. Place these scripts in outputs/inward_cycle in the reproduced PhotoQuilt project. Run from the project root in the creativegen environment.

```bash
export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
python outputs/inward_cycle/refine_spider.py --prior outputs/varied_detour_quality
python outputs/inward_cycle/check_archived.py evaluate --prior outputs/varied_detour_quality
python outputs/inward_cycle/check_archived.py select --prior outputs/varied_detour_quality
python outputs/inward_cycle/render_cycle.py --prior outputs/varied_detour_quality
```

Only the 255 visible opening leaves are eligible. Both native and 32px CLIP scores are compared with the previous output. If either is negative, generate up to three SDXL samples using the matching crop of the original pipeline-generated Iron Man latent and 12 DDIM updates. Accept a gain in both average and weaker-view margins, subject to the explicit color error bound in refine_spider.py. Attempts, prompts, steps, seeds and choices are saved in spider.json. Prompt alternatives and implementation were developed with Codex. All generation uses the official SDXL base and verification uses non-generative CLIP. The inherited results, prompts and citations are in prior.zip.

The four 16 by 16 boards share a fixed-point preview at each outgoing tile. The first three positions are inherited from the previous route. The final position is selected by comparing the opening to the fourth target and the surrounding native leaves at every position. All four camera legs zoom inward, with 5.5 s at each level. The renderer tests both physical slot pixels and the decoded GIF.

After fresh sampling the selector checks the archived same-coordinate Spider-Man candidates for any remaining weak positions. Both view margins must be positive, the average gain must exceed the recorded tolerance, and color error is limited against both the previous and original reference score. final_selection.json records the final counts.

## Rebuilding the inputs from the bundle

Start in a fresh directory containing prior.zip and inward/ from the outer archive. The commands below use the archived verifier decisions and regenerate their underlying pixels from prompts and seeds. The three raw boards of the varied run are shared with the Captain America candidate replay.

```bash
export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
unzip -q prior.zip
unzip -q prerequisites/hero_cycle.zip -d hero_source
unzip -q prerequisites/captain_refined.zip -d captain_source
mkdir -p outputs/hero_cycle outputs/varied_detour_quality outputs/inward_cycle
cp inward/*.py inward/README.md outputs/inward_cycle/
python -m sdxl.hero_cycle --output outputs/hero_cycle --recipe hero_source/sdxl/hero_cycle_recipe.json
python experiment/replay.py --output outputs/hero_cycle_varied
python - <<'PYTHON'
import json, subprocess, sys
from pathlib import Path
base = Path("outputs/hero_cycle_varied")
runs = [str((base/p).resolve()) for p in json.loads((base/"report.json").read_text())["raw_runs"]]
subprocess.run([sys.executable, "captain_source/experiment/replay.py", "--output", "outputs/captain_refined", "--runs", *runs], check=True)
PYTHON
python detour/select_route.py --base outputs/hero_cycle_varied --output outputs/varied_detour_quality
python -m sdxl --config outputs/varied_detour_quality/recipe.json --output-dir outputs/varied_detour_quality/new_board --local-files-only
cp detour/recorded/quality.json detour/recorded/spider_review.json detour/recorded/report.json outputs/varied_detour_quality/
cp prior.zip outputs/varied_detour_quality/reproduction.zip
```

Run the four inward_cycle commands above. To recompute the inherited verifier decisions as well, use detour/README.md from prior.zip. The full cold rebuild requires substantially more disk space than running the incremental experiment on already available outputs.
