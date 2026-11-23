# Revised Iron Man in the four-character Thor film

The zoom order is Spider-Man, Iron Man, Captain America, Thor, Spider-Man. This derivative is based on outward_thor16. Only Iron Man's SDXL portrait and its 256 Spider-Man trajectories are regenerated. The other three sets of native SDXL tile assets and their portal coordinates come from the approved Thor run. Their recursive thumbnail changes as the updated Iron Man passes through the loop. The exporter checks the three lossless overview images against the original outside their link rectangles and checks every reused native tile hash. The Iron Man board selects its inbound portal by the original boundary and color cost. The outbound portal into Captain America stays fixed.

Six SDXL base candidates (three prompts, two seeds per prompt) were scored. The total weights the distance to the opening Spider-Man portrait's color distribution, the actual earlier Spider-Man leaves, the local inbound and fixed outbound portal color costs, and excess yellow. See targets.json for the components, coefficients in targets.py, prompt strings and all six seeds. Each tile uses the same pose recipe, coordinate seed, and 12-step scene-conditioned sampler as the earlier Iron Man. Its local lighting varies with the new target. Final fp16 latents are compact, reproducible intermediates. The 16,384-square JPEG, 704-pixel GIF and comparison use the actual native RGB boards. The approved GIF palette, 5.5-second holds and 4.5-second outward moves are retained. The viewer-facing export includes one new native JPEG, since Iron Man is the requested replacement.

Hulk placement is a separate numerical comparison. For each of the four insertion points preserving the current sequence, hulk_placement.py estimates both directions: previous hero tiles forming Hulk, then Hulk tiles forming the next hero. It compares the 16×16 patch color distributions of the team's standalone SDXL portraits and local 4×4 color patterns in the perceptual Oklab space. Chroma receives greater weight than lightness. Sensitivity checks repeat the same analysis on the completed board overviews and on centered portrait crops at 50%, 67% and 75% of each side. These crops reduce the background contribution and are not character masks. Hulk's standalone portrait and overview come from the earlier five-hero experiment. The scores are comparative estimates and no candidate Hulk insertion is generated or included in the film.

## Reproduction

Extract reproduction.zip at the PhotoQuilt project root and use the pinned creativegen environment and local SDXL/CLIP cache. Rebuild the four-stage prerequisites using outputs/outward_thor16/README.md if their generated assets are absent. Then run:

```bash
export PYTHONPATH=. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python outputs/outward_thor_iron16/targets.py
python outputs/outward_thor_iron16/new_iron.py
python outputs/outward_thor_iron16/finish.py
```

The first two commands use one GPU. The last command checks the film, computes the four hypothetical placement scores, verifies leaf identity using frozen CLIP and bundles the dependent source/manifests. The reference PNGs used for the Hulk-only calculation are in the archive. Image generation uses official SDXL base exclusively. See prior README files for the full PhotoQuilt, SDXL, CLIP, PyTorch and Diffusers credits. Code and prompt brainstorming used Codex.
