# Joint sampling across one zoom

This experiment samples the Captain America → Hulk outward transition. The inner board depicts Captain America through Iron Man tiles. The outer board depicts Hulk through Captain America tiles. One tile of the outer board and the entire inner board share a low-frequency constraint during sampling.

## Reproduce the comparison

Use the versions in `sdxl/hero_cycle_requirements.txt` and the repository SDXL environment with the official base checkpoint cached at revision `462165984030d82259a11f4367a4eed129e94a7b`. Evaluation also needs the cached open-weight CLIP model listed below. From the repository root run

```bash
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m sdxl.joint_zoom --grid 4 --seed 31415 --slot-policy seed --output outputs/joint_zoom_pilot
python -m sdxl.evaluate_joint_zoom outputs/joint_zoom_pilot
python -m sdxl.joint_zoom --grid 4 --seed 31415 --slot-policy diffusion --output outputs/joint_zoom_diffusion
python -m sdxl.joint_zoom --grid 4 --seed 31415 --slot-policy diffusion --arm gentle --output outputs/joint_zoom_diffusion
python -m sdxl.evaluate_joint_zoom outputs/joint_zoom_diffusion
python -m unittest discover -s tests
```

Use a fresh output directory for each experiment. To resume after interruption, repeat the same command. A completed arm is recorded only after both boards are saved. For short GPU sessions, run the same command with `--arm control` and then `--arm joint`. The generator writes two prompt-generated 1024-pixel SDXL targets, two full-resolution boards and the actual sampled portal tile for each arm. The evaluator creates an animated WebP per arm. Its first transition cuts directly from the child board to the real parent crop at matching scale. The rest of the animation resizes successively wider crops of the parent. The fixed tile rasterization in generation places every decoded tile at its original latent coordinate.

## Sampling

1. The seeded baseline chooses a row-major portal index with Python Random from the run seed before generating the targets. Coordinates in the manifest start at zero. With the default `--slot-policy diffusion`, defer that choice until step 4 and use the parent tile whose early SDXL clean estimate has the smallest mean squared difference from the assembled child estimate on an 8×8 latent grid. Freeze this coordinate before applying any coupling. Both arms use the same index, the same target latents and the same initial tile noise tensor.
2. Generate both global targets with official SDXL base, 24 DDIM steps and CFG 5. Upsample each target latent and split it into spatially aligned fields. Use their low frequencies to initialize the tile states at training timestep 836. Denoise each tile at native 1024 resolution with its level's text, CFG 5.5 and 12 DDIM steps. Both arms receive the same local target guidance.
3. At steps 4 through 11 of the joint arm, compute the clean estimate of every tile before updating either scale. Assemble the child estimates in row-major order, average-pool to portal resolution and take the smooth difference from the portal estimate. The parent tile moves 85% and the children collectively 15% of the scheduled correction toward consensus. Weight each clean-estimate correction by the next DDIM signal coefficient. The final SDXL denoising step operates on both corrected states.
   The optional gentle arm uses the same position and noise, restricts the residual to an 8×8 latent grid, and applies a gain of 0.35 during steps 4–8. The last four updates preserve SDXL’s opportunity to resolve character details.
4. Decode both scales independently with the base VAE. Record mean absolute RGB error in [0,1] for the portal versus child overview and for each board versus its own target. The evaluator also checks the portal pixels against the exported parent crop and scores each board and tile against three fixed character labels with CLIP. CLIP scores are diagnostic only.

## Submission record

The run manifest includes all prompt strings, seeds, grid, model revision, environment, sample counts and GPU allocator peak. Each 4×4 arm samples 32 tiles. There are two unique target images. Separate invocations regenerate them deterministically from their seeds. The two slot policies use the same seeds. Every sample is retained. The frozen model performs inference only. All visual inputs originate from the two team-prompted SDXL calls or seeded noise in that run. Disclose Codex assistance for implementation and prompt brainstorming in the write-up. Include the experiment source, recipe, manifests, dependencies and references for any displayed result.

## Pilot results

The two 4×4 experiments used one master seed (31415), two SDXL target seeds (31416 and 31417), and tile noise seed 31418. Each of the five completed arms retained both 4096×4096 boards. One additional seeded joint attempt was interrupted during export and rerun with identical seeds. No candidate was discarded for appearance. The two independently regenerated control boards matched byte-for-byte across policies. Peak PyTorch reserved GPU memory across runs was 11.18 GiB on one GRID A100D-20C.

| Location rule (1-based) | Coupling | Portal versus child, 32×32 MAE | Parent versus Hulk target, 16×16 MAE | Child versus Captain target, 16×16 MAE |
| --- | --- | ---: | ---: | ---: |
| Seed, row 3 column 2 | Independent | .2460 | .0311 | .0224 |
| Seed, row 3 column 2 | Strong | .0131 | .0408 | .0319 |
| Step-4 prediction, row 4 column 4 | Independent | .2076 | .0311 | .0224 |
| Step-4 prediction, row 4 column 4 | Strong | .0122 | .0378 | .0296 |
| Step-4 prediction, row 4 column 4 | Gentle | .1491 | .0330 | .0236 |

At the diffusion-selected coordinate, strong guidance aligned the pose and brightness at the cut but softened the armor and workshop details. Gentle guidance kept the shield and armor clearer while leaving a noticeable change in composition at the cut. These measurements cover one seed, one transition and the 4×4 grid. The original five-character film is a separate experiment.

## References

- [PhotoQuilt paper](https://arxiv.org/abs/2606.30968) and [upstream implementation](https://github.com/KooroshRH/PhotoQuilt) for diffusion photomosaics. The experiment uses the repository's SDXL backend and explicit DDIM tail.
- [Generative Powers of Ten](https://powers-of-10.github.io/) for related work on diffusion across multiple zoom scales.
- [SDXL](https://arxiv.org/abs/2307.01952), [official SDXL base weights](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [DDIM](https://arxiv.org/abs/2010.02502).
- [CLIP](https://arxiv.org/abs/2103.00020) and [evaluation weights](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268).
- [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), [Accelerate](https://github.com/huggingface/accelerate), [PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/) and [Safetensors](https://github.com/huggingface/safetensors).
