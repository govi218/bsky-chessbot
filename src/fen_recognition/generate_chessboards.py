import random
from io import BytesIO
from pathlib import Path

import numpy as np
from cairosvg import svg2png
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pyfastnoiselite.pyfastnoiselite import (
    CellularDistanceFunction,
    CellularReturnType,
    DomainWarpType,
    FastNoiseLite,
    FractalType,
    NoiseType,
    RotationType3D,
)

from src import common, consts
from src.games import get_game
from src.render_config import (
    get_render_config,
    list_board_theme_paths,
    open_board_theme,
)


def _random_uniform_board(board_h: int, board_w: int) -> Image.Image:
    min_color = random.randint(0, 120)
    max_color = random.randint(140, 255)
    color = (
        random.randint(min_color, max_color),
        random.randint(min_color, max_color),
        random.randint(min_color, max_color),
        255,
    )
    return Image.new("RGBA", (board_w, board_h), color)


def _make_noise_generator() -> FastNoiseLite:
    noise = FastNoiseLite(seed=random.randint(0, 2**31 - 1))

    # --- Noise Type ---
    noise_type = random.choice(
        [
            NoiseType.NoiseType_OpenSimplex2,
            NoiseType.NoiseType_OpenSimplex2S,
            NoiseType.NoiseType_Cellular,
            NoiseType.NoiseType_Perlin,
            NoiseType.NoiseType_ValueCubic,
            NoiseType.NoiseType_Value,
        ]
    )
    noise.noise_type = noise_type
    noise.frequency = random.uniform(0.003, 0.06)

    # --- Rotation Type 3D ---
    noise.rotation_type_3d = random.choice(
        [
            RotationType3D.RotationType3D_None,
            RotationType3D.RotationType3D_ImproveXYPlanes,
            RotationType3D.RotationType3D_ImproveXZPlanes,
        ]
    )

    # --- Fractal Type ---
    fractal_type = random.choice(
        [
            FractalType.FractalType_None,
            FractalType.FractalType_FBm,
            FractalType.FractalType_Ridged,
            FractalType.FractalType_PingPong,
            FractalType.FractalType_DomainWarpProgressive,
            FractalType.FractalType_DomainWarpIndependent,
        ]
    )
    noise.fractal_type = fractal_type

    if fractal_type != FractalType.FractalType_None:
        noise.fractal_octaves = random.randint(2, 8)
        noise.fractal_lacunarity = random.uniform(1.5, 3.0)
        noise.fractal_gain = random.uniform(0.3, 0.7)
        noise.fractal_weighted_strength = random.uniform(0.0, 1.0)
        if fractal_type == FractalType.FractalType_PingPong:
            noise.fractal_ping_pong_strength = random.uniform(1.0, 5.0)

    # --- Cellular settings (only meaningful for Cellular noise) ---
    if noise_type == NoiseType.NoiseType_Cellular:
        noise.cellular_distance_function = random.choice(
            [
                CellularDistanceFunction.CellularDistanceFunction_Euclidean,
                CellularDistanceFunction.CellularDistanceFunction_EuclideanSq,
                CellularDistanceFunction.CellularDistanceFunction_Manhattan,
                CellularDistanceFunction.CellularDistanceFunction_Hybrid,
            ]
        )
        noise.cellular_return_type = random.choice(
            [
                CellularReturnType.CellularReturnType_CellValue,
                CellularReturnType.CellularReturnType_Distance,
                CellularReturnType.CellularReturnType_Distance2,
                CellularReturnType.CellularReturnType_Distance2Add,
                CellularReturnType.CellularReturnType_Distance2Sub,
                CellularReturnType.CellularReturnType_Distance2Mul,
                CellularReturnType.CellularReturnType_Distance2Div,
            ]
        )
        noise.cellular_jitter = random.uniform(0.2, 1.5)

    # --- Domain Warp ---
    if random.random() < 0.4:
        noise.domain_warp_type = random.choice(
            [
                DomainWarpType.DomainWarpType_OpenSimplex2,
                DomainWarpType.DomainWarpType_OpenSimplex2Reduced,
                DomainWarpType.DomainWarpType_BasicGrid,
            ]
        )
        noise.domain_warp_amp = random.uniform(10.0, 200.0)

    return noise


def _noisy_gray_board(board_h: int, board_w: int) -> Image.Image:
    noise = _make_noise_generator()
    ys, xs = np.mgrid[0:board_h, 0:board_w].astype(np.float32)
    coords = np.array([xs.reshape(-1), ys.reshape(-1)], dtype=np.float32)
    noise_array = noise.gen_from_coords(coords).reshape((board_h, board_w))
    noise_array = np.interp(
        noise_array, (noise_array.min(), noise_array.max()), (0, 255)
    ).astype(np.uint8)
    return Image.fromarray(noise_array, mode="L")


def _noisy_color_board(board_h: int, board_w: int) -> Image.Image:
    return Image.merge(
        "RGB",
        (
            _noisy_gray_board(board_h, board_w),
            _noisy_gray_board(board_h, board_w),
            _noisy_gray_board(board_h, board_w),
        ),
    )


def _svg_to_image(svg_file: Path, tile_size: int) -> Image.Image:
    with open(svg_file, "rb") as f:
        svg_data = f.read()
    png_data = svg2png(
        bytestring=svg_data, output_width=tile_size, output_height=tile_size
    )
    return Image.open(BytesIO(png_data)).convert("RGBA")


def _placeholder_piece(symbol: str, tile_size: int) -> Image.Image:
    img = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = (235, 235, 235, 255) if symbol.isupper() else (35, 35, 35, 255)
    outline = (20, 20, 20, 255) if symbol.isupper() else (220, 220, 220, 255)
    margin = max(1, tile_size // 12)
    draw.ellipse(
        (margin, margin, tile_size - margin, tile_size - margin),
        fill=fill,
        outline=outline,
        width=max(1, tile_size // 20),
    )
    font = ImageFont.load_default()
    text = symbol.upper()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (tile_size - tw) // 2
    ty = (tile_size - th) // 2
    text_fill = (15, 15, 15, 255) if symbol.isupper() else (245, 245, 245, 255)
    draw.text((tx, ty), text, fill=text_fill, font=font)
    return img


def _symbol_to_asset_key(symbol: str) -> str:
    return ("w" if symbol.isupper() else "b") + symbol.lower()


def _load_piece_images(game: str, tile_size: int) -> list[dict[str, Image.Image]]:
    spec = get_game(game)
    config = get_render_config(spec)

    all_sets: list[dict[str, Image.Image]] = []
    for piece_set in config.piece_sets:
        images: dict[str, Image.Image] = {}
        if piece_set.use_placeholders:
            for symbol in spec.piece_symbols:
                images[_symbol_to_asset_key(symbol)] = _placeholder_piece(
                    symbol, tile_size
                )
            all_sets.append(images)
            continue

        file_names = config.piece_file_names_by_provider.get(piece_set.provider, ())
        set_dir = Path(f"resources/pieces/{piece_set.provider}/{piece_set.set_name}")
        for file_name in file_names:
            path = set_dir / file_name
            if not path.exists():
                continue
            if path.suffix.lower() == ".svg":
                img = _svg_to_image(path, tile_size)
            else:
                img = Image.open(path).convert("RGBA").resize((tile_size, tile_size))
            images[path.stem.lower()] = img

        # Fill missing keys with placeholders so every game can generate data.
        for symbol in spec.piece_symbols:
            key = _symbol_to_asset_key(symbol)
            if key not in images:
                images[key] = _placeholder_piece(symbol, tile_size)

        all_sets.append(images)

    if not all_sets:
        fallback = {
            _symbol_to_asset_key(s): _placeholder_piece(s, tile_size)
            for s in spec.piece_symbols
        }
        all_sets.append(fallback)

    return all_sets


def _list_board_themes(board_h: int, board_w: int) -> list[Image.Image]:
    themes: list[Image.Image] = [
        _random_uniform_board(board_h, board_w),
        _noisy_gray_board(board_h, board_w).convert("RGBA"),
        _noisy_color_board(board_h, board_w).convert("RGBA"),
    ]
    for path in list_board_theme_paths():
        try:
            themes.append(open_board_theme(path, board_w=board_w, board_h=board_h))
        except Exception:
            continue
    return themes


def _warp_piece_image(img: Image.Image) -> Image.Image:
    w, h = img.size
    # Piecewise mesh warp gives stronger non-linear deformation.
    div_x = random.choice((2, 3))
    div_y = random.choice((2, 3))
    max_jitter = min(w, h) * random.uniform(0.0, 0.2)

    x_grid = [i * (w - 1) / div_x for i in range(div_x + 1)]
    y_grid = [i * (h - 1) / div_y for i in range(div_y + 1)]
    points: list[list[tuple[float, float]]] = []
    for gy, base_y in enumerate(y_grid):
        row: list[tuple[float, float]] = []
        for gx, base_x in enumerate(x_grid):
            is_border = gx == 0 or gx == div_x or gy == 0 or gy == div_y
            local_jitter = max_jitter * (0.7 if is_border else 1.0)
            x = np.clip(
                base_x + random.uniform(-local_jitter, local_jitter), 0.0, w - 1
            )
            y = np.clip(
                base_y + random.uniform(-local_jitter, local_jitter), 0.0, h - 1
            )
            row.append((float(x), float(y)))
        points.append(row)

    mesh = []
    for gy in range(div_y):
        for gx in range(div_x):
            left = int(round(x_grid[gx]))
            upper = int(round(y_grid[gy]))
            right = int(round(x_grid[gx + 1])) + 1
            lower = int(round(y_grid[gy + 1])) + 1
            if right <= left:
                right = min(w, left + 1)
            if lower <= upper:
                lower = min(h, upper + 1)
            if right <= left or lower <= upper:
                continue
            quad = (
                *points[gy][gx],
                *points[gy + 1][gx],
                *points[gy + 1][gx + 1],
                *points[gy][gx + 1],
            )
            mesh.append(((left, upper, right, lower), quad))

    return img.transform(
        (w, h),
        Image.Transform.MESH,
        mesh,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )


def _board_to_image(
    position: common.Position,
    board_image: Image.Image,
    tile_size: int,
    piece_images: dict[str, Image.Image],
    random_offset: int,
) -> Image.Image:

    spec = get_game(position.game)
    img = board_image.copy()
    grid = common.parse_piece_placement(position.piece_placement, spec)
    use_warped_pieces = random.random() < 0.8
    per_board_piece_images = piece_images
    if use_warped_pieces:
        # Keep one warp per piece type (e.g. all pawns share the same warp on this board).
        warped_by_type: dict[str, Image.Image] = {}
        per_board_piece_images = {}
        for key, piece_img in piece_images.items():
            piece_type = key
            if piece_type not in warped_by_type:
                warped_by_type[piece_type] = _warp_piece_image(piece_img)
            per_board_piece_images[key] = warped_by_type[piece_type]

    idx = 0
    for row in range(spec.board_rows):
        for col in range(spec.board_cols):
            piece = grid[idx]
            idx += 1
            if piece is None:
                continue

            piece_img = per_board_piece_images[_symbol_to_asset_key(piece)]
            if random.randint(0, 1) == 1:
                piece_img = ImageOps.mirror(piece_img)

            x = col * tile_size + random.randint(-random_offset, random_offset)
            y = row * tile_size + random.randint(-random_offset, random_offset)
            img.paste(piece_img, (x, y), piece_img)

    return img


def _flip_piece_colors(position: common.Position) -> common.Position:
    spec = get_game(position.game)
    tensor = common.position_to_tensor(position)
    flipped = common.flip_color_tensor(tensor, spec)
    return common.tensor_to_position(flipped, spec, side_to_move=position.side_to_move)


def _random_position(game: str) -> common.Position:
    spec = get_game(game)
    grid = [None] * spec.num_squares
    approx_num_empty = random.randint(1, spec.num_squares)
    for square in range(spec.num_squares):
        if random.randint(1, spec.num_squares) < approx_num_empty:
            continue
        grid[square] = random.choice(spec.piece_symbols)

    return common.Position(
        game=spec.key, piece_placement=common.grid_to_piece_placement(grid, spec)
    )


class BoardGenerator:
    def __init__(self, game: str, tile_size: int = consts.DEFAULT_TILE_SIZE):
        self.spec = get_game(game)
        self.tile_size = tile_size
        self.board_h, self.board_w = consts.board_pixel_size(self.spec, tile_size)
        self.random_offset = max(1, tile_size // 30)
        self.piece_image_sets = _load_piece_images(self.spec.key, tile_size)
        self.disk_themes = []
        for path in list_board_theme_paths():
            try:
                self.disk_themes.append(
                    open_board_theme(path, board_w=self.board_w, board_h=self.board_h)
                )
            except Exception:
                continue

    def generate_one(self) -> tuple[Image.Image, common.Position]:
        board_h, board_w = self.board_h, self.board_w

        # Pick a random board background: uniform, noise-gray, noise-color, or disk theme
        theme_choice = random.choice(
            ["uniform", "noise_gray", "noise_color"]
            + (["disk"] if self.disk_themes else [])
        )
        if theme_choice == "uniform":
            current_board = _random_uniform_board(board_h, board_w)
        elif theme_choice == "noise_gray":
            current_board = _noisy_gray_board(board_h, board_w).convert("RGBA")
        elif theme_choice == "noise_color":
            current_board = _noisy_color_board(board_h, board_w).convert("RGBA")
        else:
            current_board = random.choice(self.disk_themes).copy()

        if random.randint(0, 1) == 1:
            current_board = ImageOps.mirror(current_board)
        if random.randint(0, 1) == 1:
            current_board = ImageOps.flip(current_board)
        if random.randint(0, 1) == 1:
            noise = (
                _noisy_color_board(board_h, board_w)
                if random.randint(0, 1) == 1
                else _noisy_gray_board(board_h, board_w)
            )
            current_board.paste(noise, mask=_noisy_gray_board(board_h, board_w))

        piece_images = random.choice(self.piece_image_sets)
        position = _random_position(self.spec.key)
        image = _board_to_image(
            position, current_board, self.tile_size, piece_images, self.random_offset
        ).convert("RGB")

        if random.randint(0, 1) == 1:
            position = _flip_piece_colors(position)
            image = ImageOps.invert(image)

        return image, position
