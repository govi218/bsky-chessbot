BBOX_IMAGE_SIZE = 512

DEFAULT_TILE_SIZE = 32

BOARD_PIXEL_WIDTH = DEFAULT_TILE_SIZE * 8
SQUARE_SIZE = DEFAULT_TILE_SIZE


def board_pixel_size(game, tile_size=DEFAULT_TILE_SIZE):
    return (game.board_rows * tile_size, game.board_cols * tile_size)
