from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForImage2Image
from diffusers.models import VQModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from joblib.hashing import hash
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2.functional import to_pil_image
from tqdm import tqdm

from aeroblade.data import ImageFolder, read_files
from aeroblade.inversion import create_pipeline
from aeroblade.misc import device, safe_mkdir, write_config


@torch.no_grad()
def compute_reconstructions(
    ds: Dataset,
    repo_id: str,
    output_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    iterations: int = 1,
    seed: int = 1,
    batch_size: int = 1,
    num_workers: int = 1,
) -> list[Path]:
    """Compute AE reconstructions and save them in a unique directory."""
    if output_root is None and output_dir is None:
        raise ValueError("Either output_root or output_dir must be specified.")
    if output_root is not None and output_dir is not None:
        print("Ignoring output_root since output_dir is specified.")

    arg_dict = {"ds": ds, "repo_id": repo_id, "output_root": output_root, "seed": seed}
    if output_dir is None:
        # create output directory based on hashed arguments if not specified
        output_dir = output_root / hash(arg_dict) / str(iterations)

    # load files if output directory already exists, compute otherwise
    if not (
        output_dir.exists()
        and len(reconstruction_paths := read_files(output_dir)) == len(ds)
    ):
        safe_mkdir(output_dir)
        write_config(arg_dict, output_dir.parent)

        # if more than one iteration, recursively load previous iterations
        if iterations > 1:
            previous_paths = compute_reconstructions(
                ds=ds,
                repo_id=repo_id,
                output_root=output_root,
                iterations=iterations - 1,
                seed=seed,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            ds = ImageFolder(
                paths=previous_paths, transform=ds.transform, amount=ds.amount
            )

        # set up pipeline
        pipe = AutoPipelineForImage2Image.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16" if "kandinsky-2" not in repo_id else None,
        )
        pipe.enable_model_cpu_offload()

        # extract AE
        if hasattr(pipe, "vae"):
            ae = pipe.vae
            if hasattr(pipe, "upcast_vae"):
                pipe.upcast_vae()
        elif hasattr(pipe, "movq"):
            ae = pipe.movq
        ae.to(device())
        ae = torch.compile(ae)
        decode_dtype = next(iter(ae.post_quant_conv.parameters())).dtype

        # reconstruct
        generator = torch.Generator().manual_seed(seed)
        reconstruction_paths = []
        for images, paths in tqdm(
            DataLoader(ds, batch_size=batch_size, num_workers=num_workers),
            desc=f"Reconstructing with {repo_id}.",
        ):
            # normalize
            images = images.to(device(), dtype=ae.dtype) * 2.0 - 1.0

            # encode
            latents = retrieve_latents(ae.encode(images), generator=generator)

            # decode
            if isinstance(ae, VQModel):
                reconstructions = ae.decode(
                    latents.to(decode_dtype), force_not_quantize=True, return_dict=False
                )[0]
            else:
                reconstructions = ae.decode(
                    latents.to(decode_dtype), return_dict=False
                )[0]

            # de-normalize
            reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)

            # save
            for reconstruction, path in zip(reconstructions, paths):
                reconstruction_path = output_dir / f"{Path(path).stem}.png"
                to_pil_image(reconstruction).save(reconstruction_path)
                reconstruction_paths.append(reconstruction_path)
        print(f"Images saved to {output_dir}.")
    return reconstruction_paths


@torch.no_grad()
def compute_deeper_reconstructions(
    ds: Dataset,
    repo_id: str,
    output_root: Path,
    num_inference_steps: int,
    num_reconstruction_steps: int,
) -> list[Path]:
    """
    Compute reconstructions with AE and some inversion steps and save them in a
    unique directory.
    """
    # create output directory based on hashed arguments
    arg_dict = {
        "ds": ds,
        "repo_id": repo_id,
        "output_root": output_root,
        "num_inference_steps": num_inference_steps,
        "num_reconstruction_steps": num_reconstruction_steps,
    }
    output_dir = output_root / hash(arg_dict)

    # load files if output directory already exists, compute otherwise
    if not (
        output_dir.exists()
        and len(reconstruction_paths := read_files(output_dir)) == len(ds)
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_config(arg_dict, output_dir.parent)

        # set up pipeline
        pipe = create_pipeline(sd_model_ckpt=repo_id, use_blip_only=True)

        # reconstruct
        reconstruction_paths = []
        for _, paths in tqdm(
            DataLoader(ds, batch_size=1),
            desc=f"Reconstructing with {repo_id}.",
        ):
            path = paths[0]
            img = Image.open(path).convert("RGB").resize((512, 512))
            rec = pipe.compute_reconstruction(
                img,
                reconstruction_steps=num_reconstruction_steps,
                num_inference_steps=num_inference_steps,
            )

            # save
            reconstruction_path = output_dir / f"{Path(path).stem}.png"
            rec.save(reconstruction_path)
            reconstruction_paths.append(reconstruction_path)
        print(f"Images saved to {output_dir}.")
    return reconstruction_paths


def extract_patches(
    array: np.ndarray | torch.Tensor, size: int, stride: int
) -> np.ndarray | torch.Tensor:
    """
    Split 4D tensor into (overlapping) spatial patches.
    Output shape is batch_size x num_patches x num_channels x patch_size x patch_size
    """
    if isinstance(array, np.ndarray):
        is_ndarray = True
        array = torch.from_numpy(array)
    else:
        is_ndarray = False
    if array.ndim != 4:
        raise ValueError("array must be 4D.")
    patches = F.unfold(array, kernel_size=size, stride=stride).mT.reshape(
        array.shape[0], -1, array.shape[1], size, size
    )
    if is_ndarray:
        patches = patches.numpy()
    return patches
