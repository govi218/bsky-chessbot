# Import the modules
import numpy as np
import random
import os
import pyfastnoisesimd as fns
from io import BytesIO
from cairosvg import svg2png
from PIL import Image
from PIL import ImageOps
from pathlib import Path
from tqdm import tqdm

from src import consts, common
from src.games import CHESS

RANDOM_OFFSET = max(1, consts.SQUARE_SIZE // 40)

PIECE_SETS = [
    ("lichess", "alpha"),
    ("lichess", "caliente"),
    ("lichess", "california"),
    ("lichess", "cardinal"),
    ("lichess", "cburnett"),
    ("lichess", "celtic"),
    ("lichess", "chess7"),
    ("lichess", "chessnut"),
    ("lichess", "companion"),
    ("lichess", "dubrovny"),
    ("lichess", "fantasy"),
    ("lichess", "fresca"),
    ("lichess", "gioco"),
    ("lichess", "governor"),
    ("lichess", "icpieces"),
    ("lichess", "kiwen-suwi"),
    ("lichess", "kosal"),
    ("lichess", "leipzig"),
    ("lichess", "libra"),
    ("lichess", "maestro"),
    ("lichess", "merida"),
    ("lichess", "mpchess"),
    ("lichess", "pirouetti"),
    ("lichess", "pixel"),
    ("lichess", "reillycraig"),
    ("lichess", "riohacha"),
    ("lichess", "spatial"),
    ("lichess", "staunty"),
    ("lichess", "tatiana"),
    ("extra", "glass"),
    ("extra", "8_bit"),
    ("extra", "bases"),
    ("extra", "book"),
    ("extra", "bubblegum"),
    ("extra", "cases"),
    ("extra", "celtic"),
    ("extra", "chicago"),
    ("extra", "classic"),
    ("extra", "club"),
    ("extra", "condal"),
    ("extra", "dash"),
    ("extra", "eyes"),
    ("extra", "falcon"),
    ("extra", "fantasy_alt"),
    ("extra", "game_room"),
    ("extra", "gothic"),
    ("extra", "graffiti"),
    ("extra", "icy_sea"),
    ("extra", "iowa"),
    ("extra", "light"),
    ("extra", "lolz"),
    ("extra", "marble"),
    ("extra", "maya"),
    ("extra", "metal"),
    ("extra", "modern"),
    ("extra", "nature"),
    ("extra", "neo"),
    ("extra", "neon"),
    ("extra", "neo_wood"),
    ("extra", "newspaper"),
    ("extra", "ocean"),
    ("extra", "oslo"),
    ("extra", "royale"),
    ("extra", "sky"),
    ("extra", "space"),
    ("extra", "spatial"),
    ("extra", "tigers"),
    ("extra", "tournament"),
    ("extra", "vintage"),
    ("extra", "wood"),
    ("custom", "a"),
    ("custom", "b"),
    ("custom", "c"),
    ("custom", "d"),
    ("custom", "e"),
]

PIECE_FILE_NAMES = {
    "lichess": [
        "bB.svg",
        "bK.svg",
        "bN.svg",
        "bP.svg",
        "bQ.svg",
        "bR.svg",
        "wB.svg",
        "wK.svg",
        "wN.svg",
        "wP.svg",
        "wQ.svg",
        "wR.svg",
    ],
    "extra": [
        "bb.png",
        "bk.png",
        "bn.png",
        "bp.png",
        "bq.png",
        "br.png",
        "wb.png",
        "wk.png",
        "wn.png",
        "wp.png",
        "wq.png",
        "wr.png",
    ],
    "custom": [
        "bb.png",
        "bk.png",
        "bn.png",
        "bp.png",
        "bq.png",
        "br.png",
        "wb.png",
        "wk.png",
        "wn.png",
        "wp.png",
        "wq.png",
        "wr.png",
    ],
}


def getUniformRandomBoard():
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
    return Image.new(
        "RGBA", (consts.BOARD_PIXEL_WIDTH, consts.BOARD_PIXEL_WIDTH), color
    )


def getNoisyRandomGrayBoard():
    noise = fns.Noise()
    noise.noise_type = fns.NoiseType.Simplex
    noise.frequency = random.uniform(0.001, 0.06)
    # noise.seed=1234

    noise_array = noise.genAsGrid(
        shape=(consts.BOARD_PIXEL_WIDTH, consts.BOARD_PIXEL_WIDTH), start=(0, 0)
    )
    noise_array = np.interp(
        noise_array, (noise_array.min(), noise_array.max()), (0, 255)
    )
    noise_array = noise_array.astype(np.uint8)
    im = Image.fromarray(noise_array, mode="L")

    return im


def getNoisyRandomBoard():
    imgR = getNoisyRandomGrayBoard()
    imgG = getNoisyRandomGrayBoard()
    imgB = getNoisyRandomGrayBoard()

    return Image.merge("RGB", (imgR, imgG, imgB))


def getNoisyRandomBoardWithAlpha():
    imgR = getNoisyRandomGrayBoard()
    imgG = getNoisyRandomGrayBoard()
    imgB = getNoisyRandomGrayBoard()
    imgA = getNoisyRandomGrayBoard()

    return Image.merge("RGBA", (imgR, imgG, imgB, imgA))


BOARD_THEMES = [
    (None, getUniformRandomBoard),
    (None, getNoisyRandomGrayBoard),
    (None, getNoisyRandomBoard),
    ("lichess", "blue2.jpg"),
    ("lichess", "blue3.jpg"),
    ("lichess", "blue-marble.jpg"),
    ("lichess", "canvas2.jpg"),
    ("lichess", "green-plastic.png"),
    ("lichess", "grey.jpg"),
    ("lichess", "horsey.jpg"),
    ("lichess", "leather.jpg"),
    ("lichess", "maple2.jpg"),
    ("lichess", "maple.jpg"),
    ("lichess", "marble.jpg"),
    ("lichess", "metal.jpg"),
    ("lichess", "metal.orig.jpg"),
    ("lichess", "ncf-board.png"),
    ("lichess", "newspaper.png"),
    ("lichess", "olive.jpg"),
    ("lichess", "wood2.jpg"),
    ("lichess", "wood3.jpg"),
    ("lichess", "wood4.jpg"),
    ("lichess", "wood.jpg"),
    ("extra", "burled_wood.png"),
    ("extra", "christmas_alt.png"),
    ("extra", "christmas.png"),
    ("extra", "dark_wood.png"),
    ("extra", "dash.png"),
    ("extra", "glass.png"),
    ("extra", "graffiti.png"),
    ("extra", "icy_sea.png"),
    ("extra", "lolz.png"),
    ("extra", "marble.png"),
    ("extra", "metal.png"),
    ("extra", "neon.png"),
    ("extra", "newpaper.png"),
    ("extra", "parchment.png"),
    ("extra", "sand.png"),
    ("extra", "sea.png"),
    ("extra", "stone.png"),
    ("extra", "tournament.png"),
    ("extra", "walnut.png"),
    ("custom", "a.png"),
    ("custom", "b.png"),
]


# Define a function that takes an FEN string as input and returns a position object
def fen_to_board(fen):
    board = common.position_from_notation(fen, game=CHESS)
    if board is None:
        raise ValueError(f"Could not parse FEN: {fen}")
    return board


# Define a function that takes an svg file name as input and returns a numpy array of the image
def svg_to_image(svg_file):
    # Open the local .svg file as bytes
    with open(svg_file, "rb") as f:
        svg_data = f.read()

    # Convert the SVG data to PNG format
    png_data = svg2png(
        bytestring=svg_data,
        output_width=consts.SQUARE_SIZE,
        output_height=consts.SQUARE_SIZE,
    )

    # Open the PNG data as a PIL image and convert it to RGB mode
    pil_img = Image.open(BytesIO(png_data)).convert("RGBA")

    return pil_img


# Define a function that takes a position object and a dictionary of piece images as input and returns a PIL image object
def board_to_image(board, board_image, piece_images):
    board_image = board_image.copy()
    width, height = board_image.size
    assert consts.SQUARE_SIZE == height // 8
    grid = common.parse_piece_placement(board.piece_placement, CHESS)
    for square in range(0, 64):
        piece = grid[square]
        if piece:
            piece_key = ("w" if piece.isupper() else "b") + piece.lower()
            piece_image = piece_images[piece_key]
            if random.randint(0, 1) == 1:
                piece_image = ImageOps.mirror(piece_image)

            row = 7 - square // 8
            col = square % 8

            x = col * consts.SQUARE_SIZE + random.randint(-RANDOM_OFFSET, RANDOM_OFFSET)
            y = row * consts.SQUARE_SIZE + random.randint(-RANDOM_OFFSET, RANDOM_OFFSET)
            assert (consts.SQUARE_SIZE, consts.SQUARE_SIZE) == piece_image.size
            board_image.paste(piece_image, (x, y), piece_image)
    return board_image


def flip_piece_colors(board):
    grid = common.parse_piece_placement(board.piece_placement, CHESS)
    flipped_grid = []
    for piece in grid:
        if piece is None:
            flipped_grid.append(None)
        else:
            flipped_grid.append(piece.lower() if piece.isupper() else piece.upper())
    return common.Position(game=CHESS.key, piece_placement=common.grid_to_piece_placement(flipped_grid, CHESS))


def random_board():
    grid = [None] * CHESS.num_squares
    approx_num_empty_squares = random.randint(1, 64)
    piece_symbols = list(CHESS.piece_symbols)
    for square in range(CHESS.num_squares):
        if random.randint(1, 64) < approx_num_empty_squares:
            continue
        grid[square] = random.choice(piece_symbols)

    return common.Position(
        game=CHESS.key,
        piece_placement=common.grid_to_piece_placement(grid, CHESS),
    )


def generate_fen_training_data(
    num_total_out_positions=300000,
    outdir_root="resources/fen_images/generated_chessboards_fen",
    max_files_per_folder=10000,
):

    num_fens_per_combo = max(
        1, num_total_out_positions // (len(PIECE_SETS) * len(BOARD_THEMES))
    )
    print("num_fens_per_combo:", num_fens_per_combo)

    current_files_in_folder = 0
    current_outdir = None

    for piece_dir, piece_set in tqdm(PIECE_SETS):
        piece_images = {}
        # Loop through the 12 svg file names
        for file_name in PIECE_FILE_NAMES[piece_dir]:
            # Convert the svg file to a numpy array and store it in the dictionary
            path = Path(f"./resources/pieces/{piece_dir}/{piece_set}/{file_name}")
            if path.suffix == ".svg":
                img = svg_to_image(path)
            else:
                img = Image.open(path).convert("RGBA")
                img = img.resize((consts.SQUARE_SIZE, consts.SQUARE_SIZE))

            piece_key = path.stem.lower()
            piece_images[piece_key] = img

        for board_dir, board_theme in BOARD_THEMES:
            if board_dir is not None:
                board_image = Image.open(
                    f"./resources/board_themes/{board_dir}/{board_theme}"
                ).convert("RGBA")
                board_image.putalpha(255)
                board_image = board_image.resize(
                    (consts.BOARD_PIXEL_WIDTH, consts.BOARD_PIXEL_WIDTH)
                )

            # print(piece_dir, piece_set, "|", board_dir, board_theme)

            for i in range(0, num_fens_per_combo):
                if board_dir is None:
                    current_board_image = board_theme()
                else:
                    current_board_image = board_image.copy()

                    if random.randint(0, 1) == 1:
                        current_board_image = ImageOps.mirror(current_board_image)
                    if random.randint(0, 1) == 1:
                        current_board_image = ImageOps.flip(current_board_image)
                    if random.randint(0, 1) == 1:
                        noise = (
                            getNoisyRandomBoard()
                            if random.randint(0, 1) == 1
                            else getNoisyRandomGrayBoard()
                        )
                        current_board_image.paste(noise, mask=getNoisyRandomGrayBoard())

                if (
                    current_outdir is None
                    or current_files_in_folder >= max_files_per_folder
                ):
                    current_files_in_folder = 0
                    current_outdir = None
                    for i in range(0, num_total_out_positions):
                        potential_dir = outdir_root + "/" + str(i)
                        if not os.path.exists(potential_dir):
                            current_outdir = potential_dir
                            break
                    assert current_outdir is not None
                    os.makedirs(current_outdir, exist_ok=True)
                    # print("Current dir:", current_outdir)

                board = random_board()
                image = board_to_image(
                    board, current_board_image, piece_images
                ).convert("RGB")

                if random.randint(0, 1) == 1:
                    board = flip_piece_colors(board)
                    image = ImageOps.invert(image)

                fake_fen = board.fen().replace("/", "_").replace(" ", "+")
                image.save(current_outdir + "/" + fake_fen + ".png")
                current_files_in_folder += 1
