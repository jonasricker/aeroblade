"""
Evaluate deeper reconstructions (more than just AE).
"""

import argparse
from pathlib import Path

import pandas as pd
from aeroblade.data import ImageFolder
from aeroblade.distances import distance_from_config
from aeroblade.evaluation import tpr_at_max_fpr
from aeroblade.image import compute_deeper_reconstructions
from aeroblade.misc import safe_mkdir, write_config
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def main(args):
    output_dir = Path("output/03") / args.experiment_id
    safe_mkdir(output_dir)
    write_config(vars(args), output_dir)

    distances = []

    if args.precomputed_real_dist is not None:
        dirs = [args.fake_dir]
    else:
        dirs = [args.real_dir, args.fake_dir]

    # compute distances
    for dir in dirs:
        pbar = tqdm(
            desc="PROGRESS (deeper_reconstructions)",
            total=len(args.num_reconstruction_steps),
        )
        ds = ImageFolder(dir, amount=args.amount)

        # iterate over number of reconstruction steps
        for num_rec in args.num_reconstruction_steps:
            rec_paths = compute_deeper_reconstructions(
                ds,
                repo_id=args.repo_id,
                output_root=args.reconstruction_root,
                num_inference_steps=args.num_inference_steps,
                num_reconstruction_steps=num_rec,
            )
            ds_rec = ImageFolder(rec_paths)

            # iterate over distance metrics
            for dist_metric in args.distance_metrics:
                dist_dict, files = distance_from_config(
                    dist_metric,
                    spatial=False,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                ).compute(
                    ds_a=ds,
                    ds_b=ds_rec,
                )
                for dist_name, dist_tensor in dist_dict.items():
                    dist_tensor = dist_tensor.squeeze(1, 2, 3)
                    df = pd.DataFrame(
                        {
                            "dir": str(dir),
                            "num_reconstruction_steps": num_rec,
                            "distance_metric": dist_name,
                            "file": files,
                            "distance": list(dist_tensor.numpy()),
                        }
                    )
                    distances.append(df)
                pbar.update()

    distances = pd.concat(distances)

    # load precomputed real distances
    if args.precomputed_real_dist is not None:
        distances = pd.concat([distances, pd.read_pickle(args.precomputed_real_dist)])

    # compute detection results
    detection_results = []
    for (num_rec, dist_metric), group_df in distances.groupby(
        ["num_reconstruction_steps", "distance_metric"], sort=False
    ):
        y_score_real = group_df.query("dir == @args.real_dir.__str__()").distance.values
        y_score_fake = group_df.query("dir == @args.fake_dir.__str__()").distance.values
        y_score = y_score_real.tolist() + y_score_fake.tolist()
        y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)
        ap = average_precision_score(y_true=y_true, y_score=y_score)
        tpr5fpr = tpr_at_max_fpr(y_true=y_true, y_score=y_score, max_fpr=0.05)
        detection_results.append(
            {
                "fake_dir": str(args.fake_dir),
                "num_reconstruction_steps": num_rec,
                "distance_metric": dist_metric,
                "ap": ap,
                "tpr5fpr": tpr5fpr,
            }
        )
    pd.DataFrame(detection_results).to_csv(output_dir / "detection_results.csv")

    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-id", default="sd15")

    # images
    parser.add_argument("--precomputed-real-dist", type=Path)
    parser.add_argument("--real-dir", type=Path, default=Path("data/raw/real"))
    parser.add_argument(
        "--fake-dir",
        type=Path,
        default=Path(
            "data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai"
        ),
    )
    parser.add_argument("--amount", type=int, default=250)

    # reconstruction
    parser.add_argument(
        "--repo-id",
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument(
        "--num-reconstruction-steps",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 50],
    )

    # distance
    parser.add_argument(
        "--distance-metrics",
        nargs="+",
        default=[
            "lpips_vgg_-1",
        ],
    )

    # technical
    parser.add_argument(
        "--reconstruction-root", type=Path, default="data/deeper_reconstructions"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
