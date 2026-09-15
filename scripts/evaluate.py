


import argparse
import json

from ctoml.training.engine import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = evaluate_checkpoint(
            args.checkpoint, args.manifest, args.split, args.device, args.output
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"evaluate.py: {exc}\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
