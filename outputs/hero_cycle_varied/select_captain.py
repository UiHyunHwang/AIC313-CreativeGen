"""Score only the zoom destination. Every regular tile keeps the short generation prompt."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"nested_captain_8_16"))
from select_tile import rank_tiles, model_id, revision

positive = ["a close-up portrait of Captain America's masked face filling the frame, wearing a blue helmet with a white A",
            "a large clear head and shoulders portrait of Captain America, visible eyes, blue cowl and exposed chin"]
negative = ["a tiny full-body superhero standing in the distance", "a shield or star alone without a face",
            "an abstract texture without a person", "a flag hanging in an empty landscape"]
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    tiles=sorted((args.run/"tiles").glob("*.png"))
    assert len(tiles)==256
    scores=rank_tiles(tiles,positive,negative)
    args.output.write_text(json.dumps(dict(model=model_id,revision=revision,positive=positive,negative=negative,ranked=scores),indent=2))
    print(scores[0],flush=True)
