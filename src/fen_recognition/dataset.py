import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src import common, consts
from src.games import get_game


augment_transforms = torch.nn.Sequential(
    v2.RandomApply([common.AddGaussianNoise(std=0.1, scale_to_input_range=True)], p=0.4),
    v2.RandomApply([v2.ElasticTransform(alpha=30.0), v2.ElasticTransform(alpha=40.0)], p=0.4),
    v2.RandomGrayscale(p=0.4),
    v2.RandomPosterize(bits=2, p=0.2),
    v2.RandomApply([v2.ColorJitter(brightness=0.9, contrast=(0.1, 1.5), hue=0.3)], p=0.3),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3))], p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5))], p=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=10, p=0.1),
    v2.RandomEqualize(p=0.8),
)

affine_transforms = v2.RandomAffine(degrees=1.5, translate=(0.01, 0.01), scale=(0.99, 1.01), shear=1.5)


def get_default_transforms(game: str = "chess", tile_size: int = consts.DEFAULT_TILE_SIZE):
    spec = get_game(game)
    board_h, board_w = consts.board_pixel_size(spec, tile_size)
    return torch.nn.Sequential(
        v2.ToDtype(torch.float32),
        v2.Resize(size=(board_h, board_w), interpolation=v2.InterpolationMode.BICUBIC),
        common.MinMaxMeanNormalization(),
    )


# Backward compatibility with existing chess inference path.
default_transforms = get_default_transforms("chess")


class BoardPositionDataset(Dataset):
    def __init__(
        self,
        root_dir,
        game: str = "chess",
        tile_size: int = consts.DEFAULT_TILE_SIZE,
        augment_ratio=0.5,
        affine_augment_ratio=0.8,
        max=None,
        device=torch.device("cpu"),
    ):
        self.game = get_game(game)
        self.tile_size = tile_size
        self.device = device
        self.board_h, self.board_w = consts.board_pixel_size(self.game, tile_size)
        self.default_transforms = torch.nn.Sequential(
            v2.ToDtype(torch.float32),
            v2.Resize(size=(self.board_h, self.board_w), interpolation=v2.InterpolationMode.BICUBIC),
            common.MinMaxMeanNormalization(),
        )

        self.augments = torch.nn.Sequential(
            v2.RandomApply([affine_transforms], p=affine_augment_ratio),
            v2.RandomApply([augment_transforms], p=augment_ratio),
        )

        root_dir = Path(root_dir)
        assert root_dir.is_dir(), f"With root_dir = {root_dir}"

        img_list = common.glob_all_image_files_recursively(root_dir)

        self.image_files = []
        for filename in img_list:
            notation = common.normalize_position_notation(Path(filename).stem, self.game)
            if notation is not None:
                self.image_files.append(filename)
            else:
                print("WARNING: Couldn't detect ground truth notation:", filename)

        random.shuffle(self.image_files)
        if max is not None:
            self.image_files = self.image_files[0 : min(len(self.image_files), max)]

        print(f"Found {len(self.image_files)} files")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        notation = common.normalize_position_notation(file_path.stem, self.game)
        assert notation is not None

        position = common.position_from_notation(notation, self.game)
        assert position is not None

        try:
            img = Image.open(file_path)
        except RuntimeError:
            print("Error:", file_path)
            raise

        input_img = common.to_rgb_tensor(img).to(self.device)
        target = common.position_to_tensor(position).to(self.device)

        while True:
            input_img = self.augments(input_img)
            if input_img.isnan().any():
                print("WARNING: Found nan after augmentation. Trying again.")
                continue

            input_img = self.default_transforms(input_img)
            if input_img.isnan().any():
                print("WARNING: Found nan after default transform. Trying again.")
                continue
            break

        return (input_img, target)


# Backward compatibility aliases
ChessBoardDataset = BoardPositionDataset


def test_data_set(root_dir="resources/fen_images", game: str = "chess", max_data: int = 1000):
    d = BoardPositionDataset(root_dir=root_dir, game=game, max=max_data)

    for i in range(0, len(d)):
        img, target = d[i]
        assert not img.isnan().any()

        pos = common.tensor_to_position(target, game=game)
        print(pos.fen())

        pos_img = common.get_image(pos, width=img.shape[2], height=img.shape[1])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())
        ax1.imshow(img)
        ax2.imshow(pos_img)
        ax1.axis("off")
        ax2.axis("off")
        plt.show()
