# Outward loop with Hulk between Captain America and Thor

The camera starts on Spider-Man and pulls back to reveal Iron Man, Captain America, Hulk, Thor, then the opening Spider-Man. Every parent board has 16 × 16 cells. One cell holds the preceding board, so its individual tiles come into view during the outward move. The three retained sets of native tiles and their link coordinates come from outward_thor_iron16, including the revised Iron Man.

The existing Hulk and Thor portraits supply the target latents for the two new boards. generate.py uses the project's official SDXL Base checkpoint and the same 12-step hybrid sampler as outward_hulk16. It draws Captain America tiles for the Hulk portrait and Hulk tiles for the Thor portrait. Targets, per-tile prompts, coordinate seeds and digests are recorded in targets.json and generation.json. Native fp16 tile latents are stored alongside the manifests for repeat exports.

From the PhotoQuilt root in the creativegen environment, with the previous runs and model cache available:

```bash
export PYTHONPATH=. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m outputs.outward_captain_hulk_thor16.generate
python -m outputs.outward_captain_hulk_thor16.render
python -m outputs.outward_captain_hulk_thor16.verify
```

For an empty output directory, first reproduce the prior four-stage project following outputs/outward_thor_iron16/README.md, and run outputs/outward_hulk16/portraits.py for Hulk's prompt-and-seed portrait and the five-portrait target manifest. The new renderer decodes the stored tiles at 1,024 pixels per side, builds a 16,384-pixel parent board in memory, selects the two new link positions by border and pattern fit, and resolves the nested thumbnails. It exports five lossless 1,024-pixel previews, separate 4,096-pixel stills of the two new boards, and a looping 704-pixel GIF. Each stop holds for 5.5 seconds and each outward move takes 4.5 seconds. The verifier reads the encoded GIF, checks the loop boundary and confirms the revised four-stage source assets.
