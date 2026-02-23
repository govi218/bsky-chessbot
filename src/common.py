import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import v2

from src.games import CHESS, XIANGQI, GameSpec, get_game


@dataclass(frozen=True)
class Position:
    game: str
    piece_placement: str
    side_to_move: str = "w"
    castling: str = "-"
    en_passant: str = "-"
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def notation(self) -> str:
        spec = get_game(self.game)
        if spec.key == CHESS.key:
            return (
                f"{self.piece_placement} {self.side_to_move} {self.castling} "
                f"{self.en_passant} {self.halfmove_clock} {self.fullmove_number}"
            )
        if spec.key == XIANGQI.key:
            return f"{self.piece_placement} {self.side_to_move}"
        return f"{self.piece_placement} {self.side_to_move}"

    def fen(self) -> str:
        return self.notation()

    def board_fen(self) -> str:
        return self.piece_placement

    @property
    def occupied(self) -> int:
        return sum(1 for sq in parse_piece_placement(self.piece_placement, self.game) if sq is not None)


def to_rgb_tensor(img):
    if isinstance(img, Image.Image):
        img = v2.PILToTensor()(img)

    ch, h, w = img.shape
    if ch not in [1, 3, 4]:
        raise ValueError("Channel dimension must be 1, 3 or 4")
    if not img.dtype == torch.uint8:
        raise TypeError("Image must be of type uint8")
    img = img.float() / 255.0
    if ch == 4:
        img = img[:3, :, :]
    if ch == 1:
        img = img.repeat(3, 1, 1)
    return img


class MinMaxMeanNormalization(torch.nn.Module):
    def forward(self, tensor):
        min = tensor.min()
        max = tensor.max()
        if min >= max:
            return torch.zeros_like(tensor)
        tensor = (tensor - min) / (max - min)
        tensor -= tensor.mean()
        if torch.isnan(tensor).any():
            print("WARNING: Encountered NaN in input for MinMaxMeanNormalization")
            tensor = torch.zeros_like(tensor)
        assert tensor.mean().abs() < 0.0001, tensor.mean()
        assert 0.0 <= tensor.max() <= 1.0
        assert -1.0 <= tensor.min() <= 0.0
        return tensor


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0, scale_to_input_range=False):
        super().__init__()
        self.std = std
        self.mean = mean
        self.scale_to_input_range = scale_to_input_range

    def forward(self, tensor):
        std = self.std
        mean = self.mean
        if self.scale_to_input_range:
            range = tensor.max() - tensor.min()
            std *= range
            mean *= range
        return tensor + torch.randn_like(tensor) * std + mean


def pad(img: Image.Image, x, y):
    x = int(x)
    y = int(y)
    new_width = img.width + x * 2
    new_height = img.height + y * 2
    new_img = Image.new("RGB", (new_width, new_height), "white")
    new_img.paste(img.convert("RGB"), (x, y))
    return new_img


def _split_piece_placement_rows(piece_placement: str) -> list[str]:
    rows = piece_placement.strip().split("/")
    if not rows or any(row == "" for row in rows):
        raise ValueError("Invalid piece placement: empty row found")
    return rows


def parse_piece_placement(piece_placement: str, game: str | GameSpec) -> list[str | None]:
    spec = get_game(game)
    rows = _split_piece_placement_rows(piece_placement)
    if len(rows) != spec.board_rows:
        raise ValueError(
            f"Expected {spec.board_rows} rows for {spec.key}, found {len(rows)} in: {piece_placement}"
        )

    grid: list[str | None] = []
    for row in rows:
        col = 0
        i = 0
        while i < len(row):
            c = row[i]
            if c.isdigit():
                j = i
                while j < len(row) and row[j].isdigit():
                    j += 1
                count = int(row[i:j])
                if count <= 0:
                    raise ValueError(f"Invalid empty-square count in row: {row}")
                grid.extend([None] * count)
                col += count
                i = j
                continue
            if c not in spec.piece_set:
                raise ValueError(f"Invalid piece symbol '{c}' for game '{spec.key}'")
            grid.append(c)
            col += 1
            i += 1

        if col != spec.board_cols:
            raise ValueError(
                f"Expected {spec.board_cols} columns per row for {spec.key}, found {col} in row: {row}"
            )

    return grid


def grid_to_piece_placement(grid: list[str | None], game: str | GameSpec) -> str:
    spec = get_game(game)
    if len(grid) != spec.num_squares:
        raise ValueError(
            f"Expected {spec.num_squares} squares for {spec.key}, found {len(grid)}"
        )

    rows: list[str] = []
    idx = 0
    for _ in range(spec.board_rows):
        row_tokens: list[str] = []
        empty_count = 0
        for _ in range(spec.board_cols):
            piece = grid[idx]
            idx += 1
            if piece is None:
                empty_count += 1
                continue
            if piece not in spec.piece_set:
                raise ValueError(f"Invalid piece symbol '{piece}' for game '{spec.key}'")
            if empty_count:
                row_tokens.append(str(empty_count))
                empty_count = 0
            row_tokens.append(piece)
        if empty_count:
            row_tokens.append(str(empty_count))
        rows.append("".join(row_tokens))

    return "/".join(rows)


def grid_to_tensor(grid: list[str | None], game: str | GameSpec) -> torch.Tensor:
    spec = get_game(game)
    if len(grid) != spec.num_squares:
        raise ValueError(
            f"Expected {spec.num_squares} squares for {spec.key}, found {len(grid)}"
        )

    channels = len(spec.piece_symbols) + 1
    tensor = torch.zeros(spec.num_squares, channels)
    piece_to_idx = {p: i for i, p in enumerate(spec.piece_symbols)}
    empty_idx = len(spec.piece_symbols)
    for sq, piece in enumerate(grid):
        if piece is None:
            tensor[sq, empty_idx] = 1.0
            continue
        if piece not in piece_to_idx:
            raise ValueError(f"Invalid piece symbol '{piece}' for game '{spec.key}'")
        tensor[sq, piece_to_idx[piece]] = 1.0
    return tensor


def tensor_to_grid(tensor: torch.Tensor, game: str | GameSpec) -> list[str | None]:
    spec = get_game(game)
    expected_shape = [spec.num_squares, len(spec.piece_symbols) + 1]
    if list(tensor.shape) != expected_shape:
        raise ValueError(
            f"Expected tensor shape {expected_shape} for {spec.key}, found {list(tensor.shape)}"
        )

    grid: list[str | None] = []
    for sq in range(spec.num_squares):
        idx = tensor[sq].argmax().item()
        if idx == len(spec.piece_symbols):
            grid.append(None)
        else:
            grid.append(spec.piece_symbols[idx])
    return grid


def flip_color_tensor(tensor: torch.Tensor, game: str | GameSpec) -> torch.Tensor:
    spec = get_game(game)
    expected_shape = [spec.num_squares, len(spec.piece_symbols) + 1]
    if list(tensor.shape) != expected_shape:
        raise ValueError(
            f"Expected tensor shape {expected_shape} for {spec.key}, found {list(tensor.shape)}"
        )

    flipped = torch.zeros_like(tensor)
    piece_to_idx = {p: i for i, p in enumerate(spec.piece_symbols)}

    for sq in range(spec.num_squares):
        for i, symbol in enumerate(spec.piece_symbols):
            other = spec.color_swap_map[symbol]
            flipped[sq, piece_to_idx[other]] = tensor[sq, i]
        flipped[sq, len(spec.piece_symbols)] = tensor[sq, len(spec.piece_symbols)]

    return flipped


def rotate_tensor_180(tensor: torch.Tensor, game: str | GameSpec) -> torch.Tensor:
    spec = get_game(game)
    expected_shape = [spec.num_squares, len(spec.piece_symbols) + 1]
    if list(tensor.shape) != expected_shape:
        raise ValueError(
            f"Expected tensor shape {expected_shape} for {spec.key}, found {list(tensor.shape)}"
        )

    rotated = torch.zeros_like(tensor)
    for sq in range(spec.num_squares):
        row = sq // spec.board_cols
        col = sq % spec.board_cols
        mirrored_row = spec.board_rows - 1 - row
        mirrored_col = spec.board_cols - 1 - col
        mirrored_idx = mirrored_row * spec.board_cols + mirrored_col
        rotated[mirrored_idx] = tensor[sq]
    return rotated


def position_from_notation(notation: str, game: str | GameSpec) -> Position | None:
    spec = get_game(game)
    normalized = normalize_position_notation(notation, spec)
    if normalized is None:
        return None

    if spec.key == CHESS.key:
        pp, side, castling, ep, half, full = normalized.split(" ")
        return Position(
            game=spec.key,
            piece_placement=pp,
            side_to_move=side,
            castling=castling,
            en_passant=ep,
            halfmove_clock=int(half),
            fullmove_number=int(full),
        )

    pp, side = normalized.split(" ")
    return Position(game=spec.key, piece_placement=pp, side_to_move=side)


def position_to_tensor(position: Position) -> torch.Tensor:
    grid = parse_piece_placement(position.piece_placement, position.game)
    return grid_to_tensor(grid, position.game)


def tensor_to_position(
    tensor: torch.Tensor,
    game: str | GameSpec,
    side_to_move: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove_clock: int = 0,
    fullmove_number: int = 1,
) -> Position:
    spec = get_game(game)
    grid = tensor_to_grid(tensor, spec)
    return Position(
        game=spec.key,
        piece_placement=grid_to_piece_placement(grid, spec),
        side_to_move=side_to_move,
        castling=castling,
        en_passant=en_passant,
        halfmove_clock=halfmove_clock,
        fullmove_number=fullmove_number,
    )


def get_image(position_or_notation: Position | str, width: int, height: int):
    if isinstance(position_or_notation, Position):
        position = position_or_notation
    else:
        raise ValueError("Pass a Position object to get_image()")

    spec = get_game(position.game)
    grid = parse_piece_placement(position.piece_placement, spec)

    image = Image.new("RGBA", (width, height), (245, 245, 245, 255))
    draw = ImageDraw.Draw(image)

    square_w = width / spec.board_cols
    square_h = height / spec.board_rows
    dark = (181, 136, 99, 255)
    light = (240, 217, 181, 255)

    idx = 0
    for row in range(spec.board_rows):
        for col in range(spec.board_cols):
            x1 = int(col * square_w)
            y1 = int(row * square_h)
            x2 = int((col + 1) * square_w)
            y2 = int((row + 1) * square_h)
            draw.rectangle([x1, y1, x2, y2], fill=(dark if (row + col) % 2 else light))

            piece = grid[idx]
            idx += 1
            if piece is not None:
                draw.text((x1 + int(square_w * 0.35), y1 + int(square_h * 0.25)), piece, fill=(0, 0, 0, 255))

    return image


def normalize_piece_placement(
    pseudo_piece_placement: str, game: str | GameSpec
) -> str | None:
    spec = get_game(game)

    placement = pseudo_piece_placement
    placement = placement.replace("_", "/").replace("-", "/").replace(".", "1")

    try:
        grid = parse_piece_placement(placement, spec)
        return grid_to_piece_placement(grid, spec)
    except ValueError:
        return None


def normalize_position_notation(
    pseudo_notation: str, game: str | GameSpec
) -> str | None:
    spec = get_game(game)

    notation = pseudo_notation.strip()
    notation = notation.replace("+", " ")
    notation = re.sub(r"\s+", " ", notation).strip()
    if not notation:
        return None

    tokens = notation.split(" ")
    placement = normalize_piece_placement(tokens[0], spec)
    if placement is None:
        return None

    if spec.key == CHESS.key:
        side = tokens[1] if len(tokens) > 1 else "w"
        castling = tokens[2] if len(tokens) > 2 else "-"
        ep = tokens[3] if len(tokens) > 3 else "-"
        half = tokens[4] if len(tokens) > 4 else "0"
        full = tokens[5] if len(tokens) > 5 else "1"

        if side not in {"w", "b"}:
            return None
        if not re.match(r"^(-|[KQkq]{1,4})$", castling):
            return None
        if ep != "-" and not re.match(r"^[a-h][36]$", ep):
            return None
        if not half.isdigit() or not full.isdigit():
            return None

        castling = "".join(ch for ch in "KQkq" if ch in castling) or "-"
        return f"{placement} {side} {castling} {ep} {int(half)} {max(1, int(full))}"

    if spec.key == XIANGQI.key:
        side = "w"
        if len(tokens) >= 2:
            side_token = tokens[1].lower()
            if side_token not in {"w", "b", "r"}:
                return None
            side = "w" if side_token in {"w", "r"} else "b"
        return f"{placement} {side}"

    side = tokens[1] if len(tokens) >= 2 else "w"
    return f"{placement} {side}"

def glob_all_image_files_recursively(dir) -> list:
    return list(
        (
            p.resolve()
            for p in Path(dir).glob("**/*")
            if p.suffix in {".png", ".jpeg", ".jpg"}
        )
    )
