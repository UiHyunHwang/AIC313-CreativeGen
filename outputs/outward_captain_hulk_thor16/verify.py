"""Verify the encoded film and measure both new boards at the viewing scale."""
import json

import numpy as np
from PIL import Image
from outputs.outward_palette16.camera import digest
from outputs.outward_captain_hulk_thor16.render import GIF, PREVIOUS_GIF
from outputs.outward_captain_hulk_thor16.settings import HEROES, PRIOR, ROOT
from outputs.outward_thor16.verify import stops


def read(directory, name):
    return json.loads((directory / name).read_text())


def main():
    render = read(ROOT, "render.json")
    generation = read(ROOT, "generation.json")
    previous = read(PRIOR, "render.json")
    assert digest(ROOT / GIF) == render["gif"]["gif_sha256"]
    assert digest(PRIOR / PREVIOUS_GIF) == previous["gif"]["gif_sha256"]
    assert render["gif"]["order"] == [*HEROES, HEROES[0]]
    assert render["gif"]["holds"] == [[10000 * i, 5500] for i in range(5)]
    assert render["gif"]["duration_ms"] == 50020
    old_stops = stops(PRIOR / PREVIOUS_GIF)
    new_stops = stops(ROOT / GIF)
    assert len(old_stops) == 4 and len(new_stops) == 5
    old_stages = read(PRIOR, "generation.json")["stages"]
    assert generation["stages"][:3] == old_stages[:3]
    errors = {}
    for i, hero in enumerate(HEROES):
        with Image.open(ROOT / (hero + "_overview.png")) as image:
            rgb = image.convert("RGB")
            assert rgb.size == (1024, 1024)
            expected = np.asarray(rgb.resize((704, 704), Image.Resampling.LANCZOS), dtype=np.int16)
        assert expected.shape == new_stops[i].shape
        errors[hero] = float(np.abs(expected - new_stops[i]).mean())
        assert errors[hero] < 16
    result = {
        "order": render["gif"]["order"],
        "gif_sha256": render["gif"]["gif_sha256"],
        "original_gif_sha256": previous["gif"]["gif_sha256"],
        "native_grid": generation["grid"],
        "native_tile_side": generation["native_tile"],
        "generated_tiles": {s["leaf"]: len(s["tiles"]) for s in generation["stages"][3:]},
        "preserved": render["preservation"],
        "overview_to_encoded_stop_mean_abs_rgb_8bit": errors,
        "duration_ms": render["gif"]["duration_ms"],
        "dimensions": [704, 704],
        "identical_loop_boundary": render["gif"]["first_last_identical"],
        "generation_peak_gpu_gib": generation["peak_reserved_gib"],
    }
    (ROOT / "verified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
