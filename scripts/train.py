


import argparse
import json

from ctoml.config import load_config
from ctoml.training.engine import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        result = train(load_config(args.config, args.set), resume=args.resume)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"train.py: {exc}\n")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
