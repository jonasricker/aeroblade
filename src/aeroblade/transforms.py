from typing import Any, Callable, List, Sequence, Union

import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as tf
from PIL import Image
from torchvision.io import decode_jpeg, encode_jpeg


def transform_from_config(config: str) -> Callable:
    """Parse config string and return matching transform."""
    name, param = config.split("_")
    try:
        param = int(param)
    except ValueError:
        param = float(param)
    for transform in [JPEG, Crop, Blur, Noise]:
        if name == transform.__name__.lower():
            return transform(param)
    else:
        raise NotImplementedError(f"No matching transform for {config}.")


class JPEG:
    def __init__(
        self,
        quality: int | tuple[int, int],
    ) -> None:
        if isinstance(quality, int):
            quality = (quality, quality)
        self.quality = quality

    def _get_params(self, quality_min: int, quality_max: int) -> int:
        return torch.randint(low=quality_min, high=quality_max + 1, size=())

    def __call__(
        self, img: Union[torch.Tensor, Image.Image]
    ) -> Union[torch.Tensor, Image.Image]:
        quality = self._get_params(
            quality_min=self.quality[0], quality_max=self.quality[1]
        )
        t_img = img
        if not isinstance(img, torch.Tensor):
            if not F._is_pil_image(img):
                raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")
            t_img = F.to_tensor(img)
        t_img = F.convert_image_dtype(t_img, torch.uint8)

        output = decode_jpeg(encode_jpeg(t_img, quality=quality))
        output = F.convert_image_dtype(output)

        if not isinstance(img, torch.Tensor):
            output = F.to_pil_image(output, mode=img.mode)
        return output


class Crop:
    def __init__(
        self,
        factor: float | tuple[float, float],
    ) -> None:
        if isinstance(factor, float):
            factor = (factor, factor)
        self.factor = factor

    def _get_params(self, factor_min: float, factor_max: float) -> float:
        return torch.empty(1).uniform_(factor_min, factor_max).item()

    def __call__(
        self, img: Union[torch.Tensor, Image.Image]
    ) -> Union[torch.Tensor, Image.Image]:
        factor = self._get_params(factor_min=self.factor[0], factor_max=self.factor[1])
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        cropped_width, cropped_height = (
            round(width * factor),
            round(height * factor),
        )
        cropped = F.center_crop(img, output_size=(cropped_height, cropped_width))
        resized = F.resize(cropped, size=(height, width))
        return resized


class Blur(tf.GaussianBlur):
    def __init__(
        self,
        sigma: float | tuple[float, float],
        kernel_size: float = 9,
    ) -> None:
        super().__init__(kernel_size=kernel_size, sigma=sigma)


class Noise:
    def __init__(self, std: float | tuple[float]) -> None:
        if isinstance(std, float):
            std = (std, std)
        self.std = std

    def _get_params(self, std_min: float, std_max: float) -> float:
        return torch.empty(1).uniform_(std_min, std_max).item()

    def __call__(
        self, img: Union[torch.Tensor, Image.Image]
    ) -> Union[torch.Tensor, Image.Image]:
        std = self._get_params(std_min=self.std[0], std_max=self.std[1])
        t_img = img
        if not isinstance(img, torch.Tensor):
            if not F._is_pil_image(img):
                raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")
            t_img = F.to_tensor(img)

        output = torch.clamp(t_img + torch.randn(t_img.size()) * std, min=0.0, max=1.0)

        if not isinstance(img, torch.Tensor):
            output = F.to_pil_image(output, mode=img.mode)
        return output


class RandomChoiceN(tf.RandomChoice):
    def __init__(
        self,
        transforms: Sequence[Callable[..., Any]],
        p: List[float] | None = None,
        n: int = 1,
    ) -> None:
        super().__init__(transforms, p)
        self.n = n

    def forward(self, *inputs: Any) -> Any:
        indices = torch.multinomial(torch.tensor(self.p), self.n)
        transform = tf.Compose([self.transforms[idx] for idx in indices])
        return transform(*inputs)
