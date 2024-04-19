"""
Compute patch-wise distances and complexities.
"""

import argparse
from pathlib import Path

import pandas as pd
from aeroblade.high_level_funcs import compute_complexities, compute_distances
from aeroblade.image import extract_patches
from aeroblade.misc import safe_mkdir, write_config


def main(args):
    output_dir = Path("output/02") / args.experiment_id
    safe_mkdir(output_dir)
    write_config(vars(args), output_dir)

    # compute distances, eventually load precomputed distances for real images
    if args.precomputed_real_dist is not None:
        dirs = args.dirs.copy()
        dirs.remove(Path("data/raw/real"))
    else:
        dirs = args.dirs.copy()
    distances = compute_distances(
        dirs=dirs,
        transforms=args.transforms,
        repo_ids=args.repo_ids,
        distance_metrics=args.distance_metrics,
        amount=args.amount,
        reconstruction_root=args.reconstruction_root,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        spatial=True,
    )
    if args.precomputed_real_dist is not None:
        distances = pd.concat([distances, pd.read_pickle(args.precomputed_real_dist)])

    # compute complexities, eventually load precomputed complexities for real images
    if args.precomputed_real_compl is not None:
        dirs = args.dirs.copy()
        dirs.remove(Path("data/raw/real"))
    else:
        dirs = args.dirs.copy()
    complexities = compute_complexities(
        dirs=dirs,
        transforms=args.transforms,
        complexity_metrics=args.complexity_metrics,
        amount=args.amount,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.precomputed_real_compl is not None:
        complexities = pd.concat(
            [complexities, pd.read_pickle(args.precomputed_real_compl)]
        )

    def _combine_distance_and_complexity(
        row: pd.Series, complexity_metric: str
    ) -> pd.Series:
        factor = row.image_size // row.distance.shape[-1]
        patches = extract_patches(
            array=row.distance[None],
            size=args.patch_size // factor,
            stride=args.patch_stride // factor,
        )
        patch_distances = -patches.mean(axis=(2, 3, 4)).flatten()  # back to positive
        patch_complexities = complexities.query(
            "dir == @row.dir and file == @row.file and transform == @row['transform'] and complexity_metric == @complexity_metric"
        )["complexity"].item()
        out = pd.Series(
            [complexity_metric, patch_distances, patch_complexities],
            index=["complexity_metric", "distance", "complexity"],
        )
        return pd.concat([row.drop("distance"), out])

    # combine distances and complexities (of patches) to new dataframe
    combined = []
    for complexity_metric in args.complexity_metrics:
        combined.append(
            distances.apply(
                _combine_distance_and_complexity,
                axis=1,
                complexity_metric=complexity_metric,
            )
        )
    combined = pd.concat(combined)

    # store result
    categoricals = [
        "dir",
        "image_size",
        "repo_id",
        "transform",
        "distance_metric",
        "complexity_metric",
    ]
    combined[categoricals] = combined[categoricals].astype("category")
    combined.to_parquet(output_dir / "combined_dist_compl.parquet")

    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-id", default="default")

    # images
    parser.add_argument("--precomputed-real-dist", type=Path)
    parser.add_argument("--precomputed-real-compl", type=Path)
    parser.add_argument(
        "--dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/real"),
            Path("data/raw/generated/CompVis-stable-diffusion-v1-1-ViT-L-14-openai"),
            Path("data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai"),
            Path(
                "data/raw/generated/stabilityai-stable-diffusion-2-1-base-ViT-H-14-laion2b_s32b_b79k"
            ),
            Path(
                "data/raw/generated/kandinsky-community-kandinsky-2-1-ViT-L-14-openai"
            ),
            Path("data/raw/generated/midjourney-v4"),
            Path("data/raw/generated/midjourney-v5"),
            Path("data/raw/generated/midjourney-v5-1"),
        ],
    )
    parser.add_argument("--amount", type=int)
    parser.add_argument("--transforms", nargs="*", default=["clean"])

    # autoencoder
    parser.add_argument(
        "--repo-ids",
        nargs="+",
        default=[
            "CompVis/stable-diffusion-v1-1",
            "stabilityai/stable-diffusion-2-base",
            "kandinsky-community/kandinsky-2-1",
        ],
    )

    # distance
    parser.add_argument(
        "--distance-metrics",
        nargs="+",
        default=[
            "lpips_vgg_2",
        ],
    )

    # complexity
    parser.add_argument(
        "--complexity-metrics",
        nargs="+",
        default=[
            "jpeg_50",
        ],
    )
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-stride", type=int, default=64)

    # technical
    parser.add_argument(
        "--reconstruction-root", type=Path, default="data/reconstructions"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
