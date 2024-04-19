"""Create AE reconstructions."""

import argparse
from pathlib import Path

from aeroblade.data import ImageFolder
from aeroblade.image import compute_reconstructions


def main(args):
    output_dir = args.output_root / f'{args.dir.name}-{args.repo_id.replace("/", "-")}'

    ds = ImageFolder(args.dir)

    compute_reconstructions(
        ds,
        repo_id=args.repo_id,
        output_dir=output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default="debug/real")
    parser.add_argument("--repo-id", default="CompVis/stable-diffusion-v1-1")
    parser.add_argument("--output-root", type=Path, default="debug/reconstructions")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
