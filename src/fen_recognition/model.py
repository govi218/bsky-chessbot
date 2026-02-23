import torch
import torch.nn as nn

from src import consts
from src.games import get_game

from torchvision import models


def get_tile_model():
    result = models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2)
    result.fc = nn.Sequential(
        torch.nn.LazyLinear(out_features=512),
        nn.ReLU(),
    )
    return result


def get_full_img_model():
    result = models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2)
    result.fc = nn.Sequential(
        torch.nn.LazyLinear(out_features=512),
        nn.ReLU(),
    )
    return result


def get_dense_model(out_features: int):
    return nn.Sequential(
        torch.nn.LazyLinear(out_features=768),
        nn.ReLU(),
        torch.nn.LazyLinear(out_features=512),
        nn.ReLU(),
        torch.nn.LazyLinear(out_features=out_features),
    )


class BoardRec(nn.Module):
    def __init__(self, game: str, tile_size: int = consts.DEFAULT_TILE_SIZE):
        super().__init__()
        self.game = get_game(game)
        self.tile_size = tile_size
        self.board_h, self.board_w = consts.board_pixel_size(self.game, tile_size)
        self.num_squares = self.game.num_squares
        self.out_channels = len(self.game.piece_symbols) + 1

        self.tile = get_tile_model()
        self.full = get_full_img_model()
        self.dense = get_dense_model(self.out_channels)

    def forward(self, img):
        batch_size, ch, h, w = img.shape

        assert h == self.board_h
        assert w == self.board_w
        assert ch == 3

        x = img
        x = x.unfold(2, self.tile_size, self.tile_size)
        x = x.unfold(3, self.tile_size, self.tile_size)
        x = x.permute(0, 2, 3, 1, 4, 5)

        x = x.reshape(
            batch_size * self.game.board_rows * self.game.board_cols,
            ch,
            self.tile_size,
            self.tile_size,
        )

        x = self.tile(x)
        x = x.reshape(batch_size, self.num_squares, -1)

        z = self.full(img)
        z = z.reshape(batch_size, 1, -1)
        z = z.expand(-1, self.num_squares, -1)

        x = torch.cat((x, z), dim=-1)
        x = x.reshape(batch_size * self.num_squares, -1)
        x = self.dense(x)
        x = x.reshape(batch_size, self.num_squares, self.out_channels)

        return x

if __name__ == "__main__":
    model = BoardRec(game="xiangqi")
    print(torch.cuda.is_available())
