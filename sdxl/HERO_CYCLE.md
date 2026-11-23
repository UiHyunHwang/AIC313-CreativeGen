# Recursive Iron Man photomosaic

Run both commands from the PhotoQuilt repository root in the creativegen environment with the official SDXL and CLIP revisions cached. The tested package versions are in hero_cycle_requirements.txt. The torch build uses CUDA 12.4. Use the SDXL fp16 safetensors variant and CLIP revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. The SDXL revision is pinned in hero_cycle_recipe.json. A fresh output directory starts from prompts and seeds. The output directory stores each experiment as well as its results. To resume the same experiment after interruption, pass the same directory. For different parameters or software, use a new directory.

```bash
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m sdxl.hero_cycle --output outputs/reproduced_cycle
python -m sdxl.hero_cycle_render --output outputs/reproduced_cycle
```

## Generation and selection

The opening target uses the pinned official SDXL base, the original Iron Man text, seed 42 and 20 DDIM updates. Each parent supplies a 1024px target for its 16×16 grid. The runner interpolates the parent latent, injects noise and denoises each independent 1024px tile using the project's SDXL port of PhotoQuilt. The intermediate parent for each zoom is the actual selected tile's latent. All stages run sequentially on one GPU. CLIP scores the generated candidates on CPU.

For Spider-Man and Captain America, eight positions spread across the parent image's luminance distribution form a paired pilot. At each position, the short prompt and the portrait prompt use the same noise, with either 12 or 20 DDIM updates starting at timestep 836. Four standalone SDXL portraits per character serve as controls. The pilot chooses the setting with the largest average of the mean and lower-quartile joint score. The joint score is the full/32px mean CLIP margin minus 0.2 times the 4×4 RGB mean absolute error to the corresponding parent patch. The two positive prompts and four contrast prompts per character are recorded in protocol.json.

The first pass covers all 256 positions. For each of its lowest-scoring 64 tiles the pipeline tries two alternative seeds and tries a third if the best accepted score is still below the original median. Selection requires a higher joint score, a CLIP margin at least as high as the original, and RGB error at most 0.04 above the original. Each seed, prompt, position and measurement is recorded. To pick which tile to enter next, the pipeline takes the maximum mean CLIP margin across the final 256 tiles, with ties broken by ascending row-major index. This criterion measures character recognizability at two scales. It does not certify anatomy, aesthetics or subjective quality.

The final Captain America grid contains 255 generated Iron Man leaves and one link to the opening board. The return cell is reserved before sampling using 4×4 RGB distance between the opening preview and candidate areas of the Captain America target. The same whole opening board can be visited repeatedly. Every board places its 255 leaves and its link in disjoint regions of the regular grid. Integer box downsampling resolves the three cyclic links until all three 1024px thumbnails are pixel-stable. The renderer verifies equality between every linked full-resolution tile and the corresponding thumbnail of the destination board.

## Outputs and disclosure

- ironman_cycle.gif is a continuously looping zoom from the opening through all three boards. report.json records a decoded first/last pixel comparison, a comparison at all camera stops, file hashes, timings for playback, sample counts and selection statistics.
- spider/mosaic.png is Iron Man made of Spider-Man. captain/mosaic.png is Spider-Man made of Captain America. iron/mosaic.png is Captain America made of Iron Man. Each is 16384×16384. Each directory includes overview.png for a lightweight viewing copy.
- protocol.json, environment.json, pilot.json, selected.json and the per-candidate JSON files retain the grading and reproduction details. The sample count includes controls, baseline candidates and alternatives, with reused pilot tiles counted once. The SDXL bootstrap contributes one further sample.
- Global and portrait prompts were brainstormed with Codex. Code assistance, all models and upstream sources should be cited in the write-up. Generation uses SDXL base. Verification uses non-generative CLIP. Rendering uses regular-grid addressing, classical resizing and GIF palette encoding.

Method and implementation sources include [PhotoQuilt](https://arxiv.org/abs/2606.30968v2), the [upstream code](https://github.com/KooroshRH/PhotoQuilt), [SDXL](https://arxiv.org/abs/2307.01952), [SDXL base weights](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [DDIM](https://arxiv.org/abs/2010.02502), [CLIP](https://arxiv.org/abs/2103.00020), [CLIP weights](https://huggingface.co/openai/clip-vit-base-patch32), [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://pytorch.org/) and [Pillow](https://python-pillow.org/).
