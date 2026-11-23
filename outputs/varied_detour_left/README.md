# Iron Man detour

Run from the extracted bundle root, using the same environment, cached official SDXL/CLIP revisions and 20 GB GPU as the included varied-cycle reproduction. If the varied base already exists, the first command to generate it can be skipped. All coordinates reported to the viewer start at 1.

```bash
export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
python experiment/replay.py --output outputs/hero_cycle_varied
OMP_NUM_THREADS=4 python detour/select_route.py --base outputs/hero_cycle_varied --output outputs/varied_detour
python -m sdxl --config outputs/varied_detour/recipe.json --output-dir outputs/varied_detour/new_board --local-files-only
python detour/render.py --base outputs/hero_cycle_varied --output outputs/varied_detour
```

Select the return by mean RGB disagreement across the shared boundaries (65%) and 4x4 color disagreement with the SDXL Captain America target (35%). Compare all 256 positions, exclude the old return, then prefer proximity to suggested row 6 column 6 within 0.015 of the lowest cost (RGB on 0–1 scale). Select the detour by full/32px non-generative CLIP margin among positions other than the old and new return. The fourth target is the selected tile's own 1024px latent. All 256 new leaves use short user-specified prompt 'iron man', 12 tile updates and noise seed 43. The original generation and scoring references, pinned model revisions and disclosures appear in the bundled base experiment. The additional CLIP verification text and code were developed with Codex. No fine-tuning or external reference images are used.

The two destinations coexist on the same regular Captain America board. The restored old slot shows its original generated leaf. The GIF approaches the new Iron Man board, reverses its camera back to Captain America, then approaches the return slot on a different path. Source checksums, fixed-point nested references, decoded scene order, both identical Captain America stops and the exact wrap frame are checked.

This revision fixes the return to 1-based row 6 column 7 at user request and restores the original leaf at row 6 column 8. The previously generated fourth board is reused.
