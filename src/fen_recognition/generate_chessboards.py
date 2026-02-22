import os
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pyfastnoisesimd as fns
from cairosvg import svg2png
from PIL import Image, ImageOps
from tqdm import tqdm

from src import common, consts
from src.games import get_game


@dataclass(frozen=True)
class RenderAssets:
    piece_sets: list[tuple[str, str]]
    piece_file_names: dict[str, list[str]]
    board_themes: list[tuple[str | None, str | callable]]


PIECE_SETS_CHESS = [
    ("lichess", "alpha"), ("lichess", "caliente"), ("lichess", "california"), ("lichess", "cardinal"),
    ("lichess", "cburnett"), ("lichess", "celtic"), ("lichess", "chess7"), ("lichess", "chessnut"),
    ("lichess", "companion"), ("lichess", "dubrovny"), ("lichess", "fantasy"), ("lichess", "fresca"),
    ("lichess", "gioco"), ("lichess", "governor"), ("lichess", "icpieces"), ("lichess", "kiwen-suwi"),
    ("lichess", "kosal"), ("lichess", "leipzig"), ("lichess", "libra"), ("lichess", "maestro"),
    ("lichess", "merida"), ("lichess", "mpchess"), ("lichess", "pirouetti"), ("lichess", "pixel"),
    ("lichess", "reillycraig"), ("lichess", "riohacha"), ("lichess", "spatial"), ("lichess", "staunty"),
    ("lichess", "tatiana"),
    ("extra", "glass"), ("extra", "8_bit"), ("extra", "bases"), ("extra", "book"), ("extra", "bubblegum"),
    ("extra", "cases"), ("extra", "celtic"), ("extra", "chicago"), ("extra", "classic"), ("extra", "club"),
    ("extra", "condal"), ("extra", "dash"), ("extra", "eyes"), ("extra", "falcon"), ("extra", "fantasy_alt"),
    ("extra", "game_room"), ("extra", "gothic"), ("extra", "graffiti"), ("extra", "icy_sea"), ("extra", "iowa"),
    ("extra", "light"), ("extra", "lolz"), ("extra", "marble"), ("extra", "maya"), ("extra", "metal"),
    ("extra", "modern"), ("extra", "nature"), ("extra", "neo"), ("extra", "neon"), ("extra", "neo_wood"),
    ("extra", "newspaper"), ("extra", "ocean"), ("extra", "oslo"), ("extra", "royale"), ("extra", "sky"),
    ("extra", "space"), ("extra", "spatial"), ("extra", "tigers"), ("extra", "tournament"), ("extra", "vintage"),
    ("extra", "wood"),
    ("custom", "a"), ("custom", "b"), ("custom", "c"), ("custom", "d"), ("custom", "e"),
]

PIECE_FILE_NAMES_CHESS = {
    "lichess": ["bB.svg", "bK.svg", "bN.svg", "bP.svg", "bQ.svg", "bR.svg", "wB.svg", "wK.svg", "wN.svg", "wP.svg", "wQ.svg", "wR.svg"],
    "extra": ["bb.png", "bk.png", "bn.png", "bp.png", "bq.png", "br.png", "wb.png", "wk.png", "wn.png", "wp.png", "wq.png", "wr.png"],
    "custom": ["bb.png", "bk.png", "bn.png", "bp.png", "bq.png", "br.png", "wb.png", "wk.png", "wn.png", "wp.png", "wq.png", "wr.png"],
}


def _get_uniform_random_board(board_h: int, board_w: int):
    max_color = 255
    min_color = 0
    if random.randint(0, 1) == 0:
        if random.randint(0, 1) == 0:
            max_color = random.randint(0, 255)
        else:
            min_color = random.randint(0, 255)
    color = (
        random.randint(min_color, max_color),
        random.randint(min_color, max_color),
        random.randint(min_color, max_color),
        255,
    )
    return Image.new("RGBA", (board_w, board_h), color)


def _get_noisy_random_gray_board(board_h: int, board_w: int):
    noise = fns.Noise()
    noise.noise_type = fns.NoiseType.Simplex
    noise.frequency = random.uniform(0.001, 0.06)

    noise_array = noise.genAsGrid(shape=(board_h, board_w), start=(0, 0))
    noise_array = np.interp(noise_array, (noise_array.min(), noise_array.max()), (0, 255))
    noise_array = noise_array.astype(np.uint8)
    return Image.fromarray(noise_array, mode="L")


def _get_noisy_random_board(board_h: int, board_w: int):
    img_r = _get_noisy_random_gray_board(board_h, board_w)
    img_g = _get_noisy_random_gray_board(board_h, board_w)
    img_b = _get_noisy_random_gray_board(board_h, board_w)
    return Image.merge("RGB", (img_r, img_g, img_b))


def _build_board_themes(board_h: int, board_w: int):
    return [
        (None, lambda: _get_uniform_random_board(board_h, board_w)),
        (None, lambda: _get_noisy_random_gray_board(board_h, board_w)),
        (None, lambda: _get_noisy_random_board(board_h, board_w)),
        ("lichess", "blue2.jpg"), ("lichess", "blue3.jpg"), ("lichess", "blue-marble.jpg"),
        ("lichess", "canvas2.jpg"), ("lichess", "green-plastic.png"), ("lichess", "grey.jpg"),
        ("lichess", "horsey.jpg"), ("lichess", "leather.jpg"), ("lichess", "maple2.jpg"),
        ("lichess", "maple.jpg"), ("lichess", "marble.jpg"), ("lichess", "metal.jpg"),
        ("lichess", "metal.orig.jpg"), ("lichess", "ncf-board.png"), ("lichess", "newspaper.png"),
        ("lichess", "olive.jpg"), ("lichess", "wood2.jpg"), ("lichess", "wood3.jpg"),
        ("lichess", "wood4.jpg"), ("lichess", "wood.jpg"),
        ("extra", "burled_wood.png"), ("extra", "christmas_alt.png"), ("extra", "christmas.png"),
        ("extra", "dark_wood.png"), ("extra", "dash.png"), ("extra", "glass.png"), ("extra", "graffiti.png"),
        ("extra", "icy_sea.png"), ("extra", "lolz.png"), ("extra", "marble.png"), ("extra", "metal.png"),
        ("extra", "neon.png"), ("extra", "newpaper.png"), ("extra", "parchment.png"), ("extra", "sand.png"),
        ("extra", "sea.png"), ("extra", "stone.png"), ("extra", "tournament.png"), ("extra", "walnut.png"),
        ("custom", "a.png"), ("custom", "b.png"),
    ]


def get_assets_for_game(game: str, board_h: int, board_w: int) -> RenderAssets:
    spec = get_game(game)
    if spec.key == "chess":
        return RenderAssets(
            piece_sets=PIECE_SETS_CHESS,
            piece_file_names=PIECE_FILE_NAMES_CHESS,
            board_themes=_build_board_themes(board_h, board_w),
        )
    raise NotImplementedError(
        f"No render asset config for game '{spec.key}'. Add piece sets and board themes in src/fen_recognition/generate_chessboards.py"
    )


def symbol_to_piece_key(symbol: str) -> str:
    return ("w" if symbol.isupper() else "b") + symbol.lower()


def svg_to_image(svg_file: Path, tile_size: int):
    with open(svg_file, "rb") as f:
        svg_data = f.read()
    png_data = svg2png(bytestring=svg_data, output_width=tile_size, output_height=tile_size)
    return Image.open(BytesIO(png_data)).convert("RGBA")


def board_to_image(position: common.Position, board_image: Image.Image, tile_size: int, piece_images: dict[str, Image.Image], random_offset: int):
    spec = get_game(position.game)
    board_image = board_image.copy()
    grid = common.parse_piece_placement(position.piece_placement, spec)

    idx = 0
    for row in range(spec.board_rows):
        for col in range(spec.board_cols):
            piece = grid[idx]
            idx += 1
            if piece is None:
                continue
            piece_key = symbol_to_piece_key(piece)
            piece_img = piece_images[piece_key]
            if random.randint(0, 1) == 1:
                piece_img = ImageOps.mirror(piece_img)

            x = col * tile_size + random.randint(-random_offset, random_offset)
            y = row * tile_size + random.randint(-random_offset, random_offset)
            board_image.paste(piece_img, (x, y), piece_img)
    return board_image


def flip_piece_colors(position: common.Position) -> common.Position:
    spec = get_game(position.game)
    grid = common.parse_piece_placement(position.piece_placement, spec)
    tensor = common.grid_to_tensor(grid, spec)
    flipped = common.flip_color_tensor(tensor, spec)
    return common.tensor_to_position(flipped, game=spec, side_to_move=position.side_to_move)


def random_position(game: str):
    spec = get_game(game)
    grid = [None] * spec.num_squares

    approx_num_empty = random.randint(1, spec.num_squares)
    piece_symbols = list(spec.piece_symbols)
    for square in range(spec.num_squares):
        if random.randint(1, spec.num_squares) < approx_num_empty:
            continue
        grid[square] = random.choice(piece_symbols)

    return common.Position(game=spec.key, piece_placement=common.grid_to_piece_placement(grid, spec))


def generate_fen_training_data(
    num_total_out_positions=300000,
    outdir_root="resources/fen_images/generated_board_positions",
    max_files_per_folder=10000,
    game: str = "chess",
    tile_size: int = consts.DEFAULT_TILE_SIZE,
):
    spec = get_game(game)
    board_h, board_w = consts.board_pixel_size(spec, tile_size)
    random_offset = max(1, tile_size // 40)

    assets = get_assets_for_game(spec.key, board_h, board_w)

    num_positions_per_combo = max(1, num_total_out_positions // (len(assets.piece_sets) * len(assets.board_themes)))
    print("num_positions_per_combo:", num_positions_per_combo)

    current_files_in_folder = 0
    current_outdir = None

    for piece_dir, piece_set in tqdm(assets.piece_sets):
        piece_images = {}

        for file_name in assets.piece_file_names[piece_dir]:
            path = Path(f"./resources/pieces/{piece_dir}/{piece_set}/{file_name}")
            if path.suffix == ".svg":
                img = svg_to_image(path, tile_size)
            else:
                img = Image.open(path).convert("RGBA")
                img = img.resize((tile_size, tile_size))

            piece_images[path.stem.lower()] = img

        for board_dir, board_theme in assets.board_themes:
            if board_dir is not None:
                board_image = Image.open(f"./resources/board_themes/{board_dir}/{board_theme}").convert("RGBA")
                board_image.putalpha(255)
                board_image = board_image.resize((board_w, board_h))

            for _ in range(0, num_positions_per_combo):
                if board_dir is None:
                    current_board_image = board_theme()
                else:
                    current_board_image = board_image.copy()
                    if random.randint(0, 1) == 1:
                        current_board_image = ImageOps.mirror(current_board_image)
                    if random.randint(0, 1) == 1:
                        current_board_image = ImageOps.flip(current_board_image)
                    if random.randint(0, 1) == 1:
                        noise = _get_noisy_random_board(board_h, board_w) if random.randint(0, 1) == 1 else _get_noisy_random_gray_board(board_h, board_w)
                        current_board_image.paste(noise, mask=_get_noisy_random_gray_board(board_h, board_w))

                if current_outdir is None or current_files_in_folder >= max_files_per_folder:
                    current_files_in_folder = 0
                    current_outdir = None
                    for i in range(0, num_total_out_positions):
                        potential_dir = outdir_root + "/" + str(i)
                        if not os.path.exists(potential_dir):
                            current_outdir = potential_dir
                            break
                    assert current_outdir is not None
                    os.makedirs(current_outdir, exist_ok=True)

                position = random_position(spec.key)
                image = board_to_image(position, current_board_image, tile_size, piece_images, random_offset).convert("RGB")

                if random.randint(0, 1) == 1:
                    position = flip_piece_colors(position)
                    image = ImageOps.invert(image)

                fake_notation = position.fen().replace("/", "_").replace(" ", "+")
                image.save(current_outdir + "/" + fake_notation + ".png")
                current_files_in_folder += 1


def generate_position_training_data(**kwargs):
    return generate_fen_training_data(**kwargs)
