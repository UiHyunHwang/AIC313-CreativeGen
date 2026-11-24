"""Use frozen DINOv2 spatial tokens and RGB borders to reserve a zoom cell."""
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from .joint_zoom import board

MODEL = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
WEIGHTS = {"spatial": .50, "global": .15, "color": .20, "border": .15}


def neighbors(slot, grid):
    row, col = divmod(slot, grid)
    if row:
        yield slot - grid, "top"
    if row + 1 < grid:
        yield slot + grid, "bottom"
    if col:
        yield slot - 1, "left"
    if col + 1 < grid:
        yield slot + 1, "right"


def edge(image, side):
    if side == "top":
        return image[..., 0, :]
    if side == "bottom":
        return image[..., -1, :]
    if side == "left":
        return image[..., :, 0]
    return image[..., :, -1]


OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def border_cost(rgb, target, grid):
    costs = []
    for slot in range(len(rgb)):
        differences = [(edge(target, side) - edge(rgb[i], OPPOSITE[side])).abs().mean()
                       for i, side in neighbors(slot, grid)]
        costs.append(torch.stack(differences).mean())
    return torch.stack(costs)


def preview_rgb(vae, states):
    # Decode a small preview of the current clean prediction, before finishing any board.
    mini = F.interpolate(states.float(), (32, 32), mode="bilinear", align_corners=False, antialias=True)
    rgb = vae.decode(mini.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
    return (rgb.float() / 2 + .5).clamp_(0, 1)


@torch.inference_mode()
def reserve(pipe, clean, grid):
    opts = dict(revision=REVISION, local_files_only=True)
    encoder = AutoModel.from_pretrained(MODEL, **opts).to("cuda").eval()
    processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=False, **opts)
    mean, std = [torch.tensor(v, device="cuda").view(1, 3, 1, 1)
                 for v in (processor.image_mean, processor.image_std)]
    def describe(images):
        x = F.interpolate(images, (224, 224), mode="bilinear", antialias=True, align_corners=False)
        tokens = encoder(pixel_values=(x - mean) / std).last_hidden_state
        patches = tokens[:, 1:].transpose(1, 2).reshape(len(images), -1, 16, 16)
        spatial = F.adaptive_avg_pool2d(patches, 4).flatten(2)
        return F.normalize(tokens[:, 0], dim=1), F.normalize(spatial, dim=1)

    pipe.vae.to("cuda")
    banks = []
    for stage in clean:
        coarse, global_features, spatial_features = [], [], []
        for states in stage.split(8):
            rgb = preview_rgb(pipe.vae, states)
            g, s = describe(rgb)
            global_features.append(g)
            spatial_features.append(s)
            coarse.append(F.adaptive_avg_pool2d(rgb, 16))
        banks.append((torch.cat(coarse), torch.cat(global_features), torch.cat(spatial_features)))
    records = []
    for j, (rgb, g, s) in enumerate(banks):
        child = board(banks[j - 1][0], grid)
        cg, cs = describe(child)
        color = F.adaptive_avg_pool2d(child, 16)[0]
        parts = {"spatial": 1 - (s * cs).sum(1).mean(1), "global": 1 - (g * cg).sum(1),
                 "color": (rgb - color).abs().mean((1, 2, 3)), "border": border_cost(rgb, color, grid)}
        costs = sum(WEIGHTS[key] * value for key, value in parts.items())
        records.append(dict(slot=int(costs.argmin()), costs=costs.tolist(),
                            components={k: v.tolist() for k, v in parts.items()}))
    pipe.vae.to("cpu")
    return records


def guide_boundary(next_tiles, clean_tiles, child_clean, slot, grid, gain):
    """Guide a narrow strip of each adjoining tile towards the child's facing edge."""
    side = clean_tiles.shape[-1]
    child = F.avg_pool2d(board(child_clean, grid).float(), grid)[0]
    width = max(2, side // 12)
    ramp = torch.linspace(1, 0, width, device=child.device)
    for i, direction in neighbors(slot, grid):
        other = OPPOSITE[direction]
        correction = edge(child, direction) - edge(clean_tiles[i].float(), other)
        correction = F.avg_pool1d(F.pad(correction[None], (3, 3), mode="replicate"), 7, 1)[0]
        if other == "top":
            next_tiles[i, :, :width, :].add_(gain * correction[:, None, :] * ramp[:, None])
        elif other == "bottom":
            next_tiles[i, :, -width:, :].add_(gain * correction[:, None, :] * ramp.flip(0)[:, None])
        elif other == "left":
            next_tiles[i, :, :, :width].add_(gain * correction[:, :, None] * ramp)
        else:
            next_tiles[i, :, :, -width:].add_(gain * correction[:, :, None] * ramp.flip(0))


BATCH = 4


def share_latents(states, slots, grid):
    # Resolve the cyclic references simultaneously; each pass shrinks the unresolved
    # region by a grid factor. Float16 rounding fixes the remaining subpixel footprint.
    for _ in range(16):
        before = torch.stack([level[slot].clone() for level, slot in zip(states, slots)])
        after = torch.cat([F.avg_pool2d(board(states[j-1], grid).float(), grid) for j in range(len(slots))]).to(states.dtype)
        for j, slot in enumerate(slots):
            states[j, slot] = after[j]
        if torch.equal(before, after):
            return
    raise RuntimeError("Recursive latent references did not converge")


def estimate_batch(pipe, tail, z, t, cond, clean, following, reserved):
    active = [i for i in range(len(z)) if i != reserved]
    if reserved is not None:
        clean[reserved].zero_()
        following[reserved].zero_()
    for start in range(0, len(active), BATCH):
        ids = active[start:start + BATCH]
        tiles = z[ids]
        n = len(ids)
        def repeat(value):
            return value.repeat_interleave(n, dim=0)
        extra = {k: repeat(v) for k, v in cond["added_cond_kwargs"].items()}
        u, v = pipe.unet(torch.cat((tiles, tiles)), t, encoder_hidden_states=repeat(cond["encoder_hidden_states"]),
                         added_cond_kwargs=extra, return_dict=False)[0].chunk(2)
        following[ids], clean[ids] = tail.step(u + 5.5*(v-u), t, tiles)
