from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as tf
from tqdm import tqdm

from aeroblade.complexities import complexity_from_config
from aeroblade.data import ImageFolder
from aeroblade.distances import distance_from_config
from aeroblade.image import compute_reconstructions
from aeroblade.transforms import transform_from_config


def compute_distances(
    dirs: list[Path],
    transforms: list[str | Callable],
    repo_ids: list[str],
    distance_metrics: list[str],
    amount: Optional[int],
    reconstruction_root: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    compute_max: bool = True,
    **distance_kwargs,
) -> pd.DataFrame:
    """Compute distances between original and reconstructed images."""
    # set up progress bar
    pbar = tqdm(
        desc="PROGRESS (compute_distances)",
        total=len(transforms) * len(dirs) * len(repo_ids) * len(distance_metrics),
    )

    distances = []

    # iterate over transforms
    for transform_config in transforms:
        if transform_config != "clean":
            transform = tf.Compose(
                [
                    transform_from_config(transform_config)
                    if isinstance(transform_config, str)
                    else transform_config,
                    tf.ToImage(),
                    tf.ToDtype(torch.float32, scale=True),
                ]
            )

        # iterate over directories
        for dir in dirs:
            if transform_config != "clean":
                ds = ImageFolder(dir, amount=amount, transform=transform)
            else:
                ds = ImageFolder(dir, amount=amount)

            # iterate over autoencoder repo_ids
            for repo_id in repo_ids:
                rec_paths = compute_reconstructions(
                    ds,
                    repo_id=repo_id,
                    output_root=reconstruction_root,
                    seed=seed,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                ds_rec = ImageFolder(rec_paths)

                # iterate over distance metrics
                for dist_metric in distance_metrics:
                    dist_dict, files = distance_from_config(
                        dist_metric,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        **distance_kwargs,
                    ).compute(
                        ds_a=ds,
                        ds_b=ds_rec,
                    )
                    for dist_name, dist_tensor in dist_dict.items():
                        if not distance_kwargs.get("spatial", False):
                            dist_tensor = dist_tensor.squeeze(1, 2, 3)
                        df = pd.DataFrame(
                            {
                                "dir": str(dir),
                                "image_size": int(ds[0][0].shape[-1]),
                                "repo_id": repo_id,
                                "transform": transform_config,
                                "distance_metric": dist_name,
                                "file": files,
                                "distance": list(dist_tensor.numpy()),
                            }
                        )
                        distances.append(df)
                    pbar.update()

    distances = pd.concat(distances)

    # determine maximum distance over all repo_ids for each file
    if compute_max:
        maxima = []
        for group_keys, group_df in distances.groupby(
            group_cols := ["dir", "image_size", "transform", "distance_metric"],
            sort=False,
        ):
            max_values = group_df.groupby("file").apply(
                lambda df: np.stack(df.distance).max(axis=0)
            )
            max_df = {col: key for col, key in zip(group_cols, group_keys)}
            max_df.update(
                {
                    "repo_id": "max",
                    "file": max_values.index.values,
                    "distance": max_values.values,
                }
            )
            maxima.append(pd.DataFrame(max_df))

        distances = pd.concat([distances, *maxima]).sort_values("dir", kind="stable")
    distances = distances.reset_index(drop=True)
    return distances


def compute_complexities(
    dirs: list[Path],
    transforms: list[str],
    complexity_metrics: list[str],
    amount: Optional[int],
    patch_size: Optional[int],
    patch_stride: Optional[int],
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Compute distances between original and reconstructed images."""
    # set up progress bar
    pbar = tqdm(
        desc="PROGRESS (compute_complexities)",
        total=len(transforms) * len(dirs) * len(complexity_metrics),
    )

    complexities = []

    # iterate over transforms
    for transform_config in transforms:
        if transform_config != "clean":
            transform = tf.Compose(
                [
                    transform_from_config(transform_config),
                    tf.ToImage(),
                    tf.ToDtype(torch.float32, scale=True),
                ]
            )

        # iterate over directories
        for dir in dirs:
            if transform_config != "clean":
                ds = ImageFolder(dir, amount=amount, transform=transform)
            else:
                ds = ImageFolder(dir, amount=amount)

            # iterate over complexity metrics
            for comp_metric in complexity_metrics:
                comp_dict, files = complexity_from_config(
                    comp_metric,
                    patch_size=patch_size,
                    patch_stride=patch_stride,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ).compute(
                    ds=ds,
                )
                for comp_name, comp_tensor in comp_dict.items():
                    df = pd.DataFrame(
                        {
                            "dir": str(dir),
                            "transform": transform_config,
                            "complexity_metric": comp_name,
                            "file": files,
                            "complexity": list(comp_tensor.numpy()),
                        }
                    )
                    complexities.append(df)
                pbar.update()

    complexities = pd.concat(complexities).reset_index(drop=True)
    return complexities
