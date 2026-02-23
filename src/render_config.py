from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.games import GameSpec, get_game


@dataclass(frozen=True)
class PieceSetConfig:
    provider: str
    set_name: str
    use_placeholders: bool = False


@dataclass(frozen=True)
class GameRenderConfig:
    piece_sets: tuple[PieceSetConfig, ...]
    piece_file_names_by_provider: dict[str, tuple[str, ...]]


CHESS_RENDER_CONFIG = GameRenderConfig(
    piece_sets=(
        PieceSetConfig("lichess", "alpha"),
        PieceSetConfig("lichess", "caliente"),
        PieceSetConfig("lichess", "cburnett"),
        PieceSetConfig("lichess", "dubrovny"),
        PieceSetConfig("lichess", "pixel"),
        PieceSetConfig("extra", "glass"),
        PieceSetConfig("extra", "neo"),
        PieceSetConfig("custom", "a"),
    ),
    piece_file_names_by_provider={
        "lichess": ("bB.svg", "bK.svg", "bN.svg", "bP.svg", "bQ.svg", "bR.svg", "wB.svg", "wK.svg", "wN.svg", "wP.svg", "wQ.svg", "wR.svg"),
        "extra": ("bb.png", "bk.png", "bn.png", "bp.png", "bq.png", "br.png", "wb.png", "wk.png", "wn.png", "wp.png", "wq.png", "wr.png"),
        "custom": ("bb.png", "bk.png", "bn.png", "bp.png", "bq.png", "br.png", "wb.png", "wk.png", "wn.png", "wp.png", "wq.png", "wr.png"),
    },
)


PLACEHOLDER_RENDER_CONFIG = GameRenderConfig(
    piece_sets=(PieceSetConfig("placeholder", "default", use_placeholders=True),),
    piece_file_names_by_provider={},
)


GAME_RENDER_CONFIGS: dict[str, GameRenderConfig] = {
    "chess": CHESS_RENDER_CONFIG,
    "xiangqi": PLACEHOLDER_RENDER_CONFIG,
    "shogi": PLACEHOLDER_RENDER_CONFIG,
}


def get_render_config(game: str | GameSpec) -> GameRenderConfig:
    spec = get_game(game)
    return GAME_RENDER_CONFIGS.get(spec.key, PLACEHOLDER_RENDER_CONFIG)


def list_board_theme_paths() -> list[Path]:
    roots = [
        Path("resources/board_themes/lichess"),
        Path("resources/board_themes/extra"),
        Path("resources/board_themes/custom"),
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                paths.append(p)
    return paths


def open_board_theme(path: Path, board_w: int, board_h: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    img.putalpha(255)
    return img.resize((board_w, board_h))
