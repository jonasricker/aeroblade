import abc
import warnings
from pathlib import Path
from typing import Any, Optional

import lpips
import pyiqa
import torch
import torch.nn.functional as F
from joblib.memory import Memory
from torch.utils.data import DataLoader
from tqdm import tqdm

from aeroblade.data import ImageFolder
from aeroblade.misc import device

mem = Memory(location="cache", compress=("lz4", 9), verbose=0)


class Distance(abc.ABC):
    """Base class for all distance metrics."""

    @torch.no_grad()
    def compute(
        self,
        ds_a: ImageFolder,
        ds_b: ImageFolder,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """
        Compute distance between two datasets with matching filenames.
        """
        files_a = [Path(f).name for f in ds_a.img_paths]
        files_b = [Path(f).name for f in ds_b.img_paths]
        if files_a != files_b:
            files_a_stems = [Path(f).stem for f in ds_a.img_paths]
            files_b_stems = [Path(f).stem for f in ds_b.img_paths]
            if files_a_stems != files_b_stems:
                raise ValueError("ds_a and ds_b should contain matching files.")
            else:
                warnings.warn(
                    "ds_a and ds_b contain files with different file endings. Make sure that is does not cause issues, e.g., you should not have different images with the same name but different file endings."
                )

        result = self._compute(
            ds_a=ds_a,
            ds_b=ds_b,
        )
        return self._postprocess(result), files_a

    @abc.abstractmethod
    def _compute(self, ds_a: ImageFolder, ds_b: ImageFolder) -> Any:
        """Distance-specific computation."""
        pass

    @abc.abstractmethod
    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        """Post-processing step that maps result into dictionary."""
        pass


class _PatchedLPIPS(lpips.LPIPS):
    """Patched version of LPIPS which returns layer-wise output without upsampling."""

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if (
            normalize
        ):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1))
            if self.version == "0.1"
            else (in0, in1)
        )
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = (
                lpips.normalize_tensor(outs0[kk]),
                lpips.normalize_tensor(outs1[kk]),
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res_no_up = [self.lins[kk](diffs[kk]) for kk in range(self.L)]
                res = [
                    lpips.upsample(res_no_up[kk], out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    lpips.spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
                    for kk in range(self.L)
                ]
                res_no_up = res
        else:
            if self.spatial:
                res_no_up = [diffs[kk].sum(dim=1, keepdim=True) for kk in range(self.L)]
                res = [
                    lpips.upsample(res_no_up[kk], out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    lpips.spatial_average(
                        diffs[kk].sum(dim=1, keepdim=True), keepdim=True
                    )
                    for kk in range(self.L)
                ]
                res_no_up = res

        val = 0
        for layer in range(self.L):
            val += res[layer]

        if retPerLayer:
            return (val, res_no_up)
        else:
            return val


@mem.cache(ignore=["batch_size", "num_workers"])
def _compute_lpips(
    ds_a: ImageFolder,
    ds_b: ImageFolder,
    model_kwargs: dict,
    batch_size: int,
    num_workers: int,
):
    dl_a = DataLoader(dataset=ds_a, batch_size=batch_size, num_workers=num_workers // 2)
    dl_b = DataLoader(dataset=ds_b, batch_size=batch_size, num_workers=num_workers // 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _PatchedLPIPS(spatial=True, **model_kwargs).to(device())

    torch.compile(model.net)

    lpips_layers = [[] for _ in range(1 + len(model.chns))]
    for (tensor_a, _), (tensor_b, _) in tqdm(
        zip(dl_a, dl_b),
        desc="Computing LPIPS",
        total=len(dl_a),
    ):
        sum_batch, layers_batch = model(
            tensor_a.to(device()),
            tensor_b.to(device()),
            retPerLayer=True,
            normalize=True,
        )
        lpips_layers[0].append(sum_batch.to(device="cpu", dtype=torch.float16))
        for i, layer_result in enumerate(layers_batch):
            lpips_layers[i + 1].append(
                layer_result.to(device="cpu", dtype=torch.float16)
            )

    lpips_layers = [torch.cat(lpips_layer) for lpips_layer in lpips_layers]
    return lpips_layers


class LPIPS(Distance):
    """From Zhang et al., The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, 2018"""

    def __init__(
        self,
        net: str = "vgg",
        layer: int = -1,
        spatial: bool = False,
        output_size: Optional[int] = None,
        concat_layers_and_flatten: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        """
        net: backbone to use from ['alex', 'vgg', 'squeeze']
        layer: layer to return, -1 returns all layers
        spatial: whether to return scores for each patch
        output_size: resize output to this size (only applicable if spatial=True)
        """
        self.net = net
        self.layer = layer
        self.spatial = spatial
        self.output_size = output_size
        self.concat_layers_and_flatten = concat_layers_and_flatten
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _compute(self, ds_a: ImageFolder, ds_b: ImageFolder) -> list[torch.Tensor]:
        """Use pure function to enable caching."""
        return _compute_lpips(
            ds_a=ds_a,
            ds_b=ds_b,
            model_kwargs={"net": self.net},
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _postprocess(self, result: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Handle layer selection and resizing."""
        out = {}
        if self.layer == -1:
            for i, tensor in enumerate(result):
                out[f"lpips_{self.net}_{i}"] = -tensor
        else:
            out[f"lpips_{self.net}_{self.layer}"] = -result[self.layer]

        for layer, tensor in out.items():
            if not self.spatial:
                out[layer] = tensor.mean((2, 3), keepdim=True)
            elif self.output_size is not None:
                out[layer] = F.interpolate(
                    tensor.to(dtype=torch.float32),
                    size=self.output_size,
                    mode="bilinear",
                    antialias=True,
                ).to(dtype=torch.float16)

        if (
            self.concat_layers_and_flatten
            and self.layer == -1
            and self.spatial
            and self.output_size is not None
        ):
            out = {
                f"lpips_{self.net}_flat": torch.cat(
                    [tensor.flatten(start_dim=1) for tensor in out.values()], dim=1
                )
            }

        return out


@mem.cache(ignore=["batch_size", "num_workers"])
def _compute_pyiqa_distance(
    ds_a: ImageFolder,
    ds_b: ImageFolder,
    metric_name: str,
    batch_size: int,
    num_workers: int,
    **metric_kwargs,
):
    dl_a = DataLoader(dataset=ds_a, batch_size=batch_size, num_workers=num_workers // 2)
    dl_b = DataLoader(dataset=ds_b, batch_size=batch_size, num_workers=num_workers // 2)

    metric = pyiqa.create_metric(metric_name, **metric_kwargs)

    out = []
    for (tensor_a, _), (tensor_b, _) in tqdm(
        zip(dl_a, dl_b),
        desc=f"Computing {metric_name}",
        total=len(dl_a),
    ):
        out_tensor = metric(tensor_a, tensor_b).to(device="cpu", dtype=torch.float16)
        if out_tensor.ndim == 0:
            out_tensor = out_tensor.unsqueeze(0)
        out.append(out_tensor)
    return torch.cat(out)


class PyIQADistance(Distance):
    def __init__(
        self,
        metric_name: str,
        batch_size: int,
        num_workers: int,
        **metric_kwargs,
    ) -> None:
        self.metric_name = metric_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metric_kwargs = metric_kwargs

    def _compute(self, ds_a: ImageFolder, ds_b: ImageFolder) -> Any:
        return _compute_pyiqa_distance(
            ds_a=ds_a,
            ds_b=ds_b,
            metric_name=self.metric_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # **self.metric_kwargs,  # this should be done on case by case basis
        )

    def _postprocess(self, result: Any) -> dict[str, torch.Tensor]:
        if pyiqa.DEFAULT_CONFIGS[self.metric_name].get("lower_better", False):
            result *= -1
        result = result[(...,) + (None,) * (4 - result.ndim)]  # make sure output is 4D
        return {self.metric_name: result}


def distance_from_config(
    config: str,
    batch_size: int = 1,
    num_workers: int = 1,
    **kwargs,
) -> Distance:
    """Parse config string and return matching distance."""
    if config.startswith("lpips"):
        _, net, layer = config.split("_")
        distance = LPIPS(
            net=net,
            layer=int(layer),
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
    else:
        distance = PyIQADistance(
            metric_name=config,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
    return distance
