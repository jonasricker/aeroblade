"""Generate images from prompts."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from aeroblade.misc import device, safe_mkdir
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True


def main(args):
    output_dir = (
        args.output_root / f'{args.repo_id.replace("/", "-")}-{args.prompt_file.stem}'
    )
    safe_mkdir(output_dir)

    # set up pipeline
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.repo_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None,
    ).to(device())
    if "kandinsky-2" not in args.repo_id:
        pipe.enable_model_cpu_offload()

    # load prompts
    prompts = pd.read_csv(args.prompt_file, dtype={"image_id": str})

    # fix generator for reproducibity
    generator = torch.Generator().manual_seed(args.seed)

    # generate and save images
    prompt_batches = [
        prompts.iloc[i : i + args.batch_size]
        for i in range(0, len(prompts), args.batch_size)
    ]
    for prompt_batch in tqdm(prompt_batches):
        image_batch = pipe(
            prompt=prompt_batch["prompt"].tolist(),
            generator=generator,
        ).images

        for image, image_id in zip(image_batch, prompt_batch["image_id"]):
            image.save(output_dir / f"{image_id}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-file", type=Path, default=Path("debug/debug_prompts.csv")
    )
    parser.add_argument("--output-root", type=Path, default="debug/generated")
    parser.add_argument(
        "--repo-id",
        choices=[
            "kandinsky-community/kandinsky-2-1",
            "stabilityai/stable-diffusion-2-1-base",
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-1",
        ],
        default="CompVis/stable-diffusion-v1-1",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
