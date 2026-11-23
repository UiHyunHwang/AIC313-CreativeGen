from pathlib import Path
from outputs.outward_palette16 import common as core
from outputs.outward_palette16.targets import config, open_rgb

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
PRIOR = OUT / "outward_thor_iron16"
HEROES = ("spider_man", "iron_man", "captain_america", "hulk", "thor")

def select_targets():
    four = core.read_json(PRIOR / "targets.json")
    five = core.read_json(OUT / "outward_hulk16" / "targets.json")
    targets = {"heroes": HEROES}
    for field in ("selected", "hashes", "latent_hashes"):
        targets[field] = [*four[field][:3], five[field][4], four[field][3]]
    core.write_json(ROOT / "targets.json", targets)
    return targets
