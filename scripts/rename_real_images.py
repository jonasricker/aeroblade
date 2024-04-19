"""Rename LAION-5B images s.t. filenames correspond to prompt IDs."""

import argparse
from pathlib import Path

import pandas as pd
from aeroblade.misc import safe_mkdir
from tqdm import tqdm


def main(args):
    safe_mkdir(args.output_dir)
    metadata = pd.read_parquet(args.metadata_path)
    image_files = sorted(args.image_dir.glob("*.png"))
    for file in tqdm(image_files, desc="Renaming files"):
        idx = int(file.stem)
        new_idx = metadata.index[idx]
        file.rename(args.output_dir / f"{new_idx:09}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-path", type=Path, default="data/raw/real/real_metadata.parquet"
    )
    parser.add_argument("--image-dir", type=Path, default="tmp/laion/00000")
    parser.add_argument("--output-dir", type=Path, default="data/raw/real")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
