import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset, TensorDataset
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes

from src.common import to_rgb_tensor, MinMaxMeanNormalization, AddGaussianNoise
from src import consts
from src.bounding_box.generate_chessboards_bbox import BboxGenerator


def box_to_mask(box: torch.Tensor):
    # box should be in x1, y1, x2, y2 format and the values should be relative
    mask = torch.zeros([consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE])
    x1, y1, x2, y2 = box.to(int)
    mask[y1:y2, x1:x2] = 1.0
    return mask.unsqueeze(0)


default_transforms = torch.nn.Sequential(
    v2.ToDtype(torch.float32),
    v2.Resize(
        size=(consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE),
        interpolation=v2.InterpolationMode.BICUBIC,
    ),
    MinMaxMeanNormalization(),
)

augment_transforms = torch.nn.Sequential(
    v2.RandomApply([v2.RandomAffine(degrees=1.0, shear=1.0)], p=0.3),
    v2.RandomInvert(p=0.1),
    v2.RandomApply([AddGaussianNoise(std=0.1, scale_to_input_range=True)], p=0.4),
    v2.RandomApply(
        [v2.ElasticTransform(alpha=30.0), v2.ElasticTransform(alpha=40.0)], p=0.4
    ),
    v2.RandomGrayscale(p=0.4),
    v2.RandomPosterize(bits=2, p=0.2),
    v2.RandomApply(
        [v2.ColorJitter(brightness=0.9, contrast=(0.1, 1.5), hue=0.3)], p=0.3
    ),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3))], p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5))], p=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=10, p=0.1),
    v2.RandomEqualize(p=0.8),
)


class GenerativeBboxDataset(IterableDataset):
    def __init__(
        self,
        game: str,
        augment_ratio: float = 0.5,
    ):
        self.generator = BboxGenerator(game)
        self.augment_ratio = augment_ratio

    def __iter__(self):
        while True:
            image, (center_x, center_y, width, height) = self.generator.generate_one()
            input_img = to_rgb_tensor(image)

            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            box = torch.tensor([x1, y1, x2, y2])

            do_augment = self.augment_ratio > random.uniform(0, 1)

            while True:
                if do_augment:
                    input_img = augment_transforms(input_img)
                if input_img.isnan().any():
                    print("WARNING: Found nan after augmentation. Trying again.")
                    continue
                input_img = default_transforms(input_img)
                if input_img.isnan().any():
                    print("WARNING: Found nan after default transform. Trying again.")
                    continue
                break

            yield input_img, box, box_to_mask(box)


def generate_fixed_test_set(
    game: str,
    size: int = 500,
    seed: int = 42,
) -> TensorDataset:
    rng_state = random.getstate()
    random.seed(seed)

    generator = BboxGenerator(game)
    images = []
    boxes = []
    masks = []
    for _ in range(size):
        image, (center_x, center_y, width, height) = generator.generate_one()
        input_img = to_rgb_tensor(image)
        input_img = default_transforms(input_img)

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        box = torch.tensor([x1, y1, x2, y2])

        images.append(input_img)
        boxes.append(box)
        masks.append(box_to_mask(box))

    random.setstate(rng_state)
    return TensorDataset(torch.stack(images), torch.stack(boxes), torch.stack(masks))


def test_data_set(game: str, size: int = 1000):
    ds = GenerativeBboxDataset(game=game, augment_ratio=0.5)

    for i, (img, target_box, target_mask) in enumerate(ds):
        if i >= size:
            break
        assert not img.isnan().any()

        vis = img.clone()
        vis *= 255.0
        vis += 128.0
        vis = vis.to(torch.uint8)
        vis = draw_bounding_boxes(vis, target_box.unsqueeze(0), width=5, colors="red")

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow((vis.permute(1, 2, 0) - vis.min()) / (vis.max() - vis.min()))
        ax2.imshow(target_mask.squeeze(0))
        plt.show()
