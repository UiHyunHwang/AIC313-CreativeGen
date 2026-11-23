# Five-stage outward SDXL cycle

Spider-Man pulls out into Iron Man, Captain America, Thor, Hulk and back into the original Spider-Man portrait. Each 16×16 level has 256 native 1024-pixel SDXL leaves and one automatic recursive link when rendered. A 5.5-second hold precedes each 4.5-second move. The final hold image and the opening frame match at decoded GIF pixel precision.

Iron Man, Captain America and Thor reuse their prior generation records and all native source tiles. Their previous linked coordinates stay fixed. The portal thumbnail is recalculated for the five-stage graph. At render time all ordinary native source assets and the completed lossless overviews outside these cells are checked against the earlier run. The four-stage GIF remains a separate deliverable.

Hulk's target portrait is selected among four official SDXL base samples at 24 steps, using the color fit to Thor, the opening Spider-Man and the previously generated Thor leaves. Seeds and all candidate scores are in targets.json. The new Spider-Man grid uses Hulk leaves; the new Hulk grid uses Thor leaves. The two schedules and lighting choices match the previous four-stage experiment. Each of the 512 affected cells uses one trajectory at 12 steps and its own recorded seed. Final native fp16 latents are stored as safetensors and decoded by the same official base VAE for rendering. This avoids duplicate lossless tile PNGs on disk. New and inherited native leaves are assembled before the recursive positions are selected.

The original GIF palette has few useful olive tones. On the new scenes, 32 low-impact palette entries are reassigned using greens measured in the new lossless board previews. The three inherited stops use the original palette. At their boundaries, the palette follows the outgoing board's share of the frame. The verification report compares the delivered Hulk stop with its pre-quantization RGB image and the previous palette. Frozen CLIP is used only for an audit at the actual 44-pixel display size. No training, external reference images, manual painting or rerolls of completed tiles are used.

## Reproduction

Extract reproduction.zip at the PhotoQuilt project root. It includes the prior four-stage source and prerequisites. If those results are absent, reproduce that run using outputs/outward_thor16/README.md. With the pinned creativegen environment and the official weights cached locally, run:

```bash
export PYTHONPATH=. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python outputs/outward_hulk16/portraits.py
python outputs/outward_hulk16/new_boards.py
python outputs/outward_hulk16/render_cycle.py
python outputs/outward_hulk16/verify.py
```

The first three commands use one GPU. Source manifests, prompts, seeds, scores and hashes accompany the code. Output JPEGs measure 16,384 × 16,384. The previous experiment's README gives credits to PhotoQuilt, Podell et al. SDXL, Radford et al. CLIP, Diffusers and PyTorch. Code and prompt brainstorming used Codex.
