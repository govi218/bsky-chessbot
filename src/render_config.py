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
        PieceSetConfig("lichess", "california"),
        PieceSetConfig("lichess", "cardinal"),
        PieceSetConfig("lichess", "cburnett"),
        PieceSetConfig("lichess", "celtic"),
        PieceSetConfig("lichess", "chess7"),
        PieceSetConfig("lichess", "chessnut"),
        PieceSetConfig("lichess", "companion"),
        PieceSetConfig("lichess", "dubrovny"),
        PieceSetConfig("lichess", "fantasy"),
        PieceSetConfig("lichess", "fresca"),
        PieceSetConfig("lichess", "gioco"),
        PieceSetConfig("lichess", "governor"),
        PieceSetConfig("lichess", "icpieces"),
        PieceSetConfig("lichess", "kiwen-suwi"),
        PieceSetConfig("lichess", "kosal"),
        PieceSetConfig("lichess", "leipzig"),
        PieceSetConfig("lichess", "libra"),
        PieceSetConfig("lichess", "maestro"),
        PieceSetConfig("lichess", "merida"),
        PieceSetConfig("lichess", "mpchess"),
        PieceSetConfig("lichess", "pirouetti"),
        PieceSetConfig("lichess", "pixel"),
        PieceSetConfig("lichess", "reillycraig"),
        PieceSetConfig("lichess", "riohacha"),
        PieceSetConfig("lichess", "spatial"),
        PieceSetConfig("lichess", "staunty"),
        PieceSetConfig("lichess", "tatiana"),
        PieceSetConfig("extra", "8_bit"),
        PieceSetConfig("extra", "bases"),
        PieceSetConfig("extra", "book"),
        PieceSetConfig("extra", "bubblegum"),
        PieceSetConfig("extra", "cases"),
        PieceSetConfig("extra", "celtic"),
        PieceSetConfig("extra", "chicago"),
        PieceSetConfig("extra", "classic"),
        PieceSetConfig("extra", "club"),
        PieceSetConfig("extra", "condal"),
        PieceSetConfig("extra", "dash"),
        PieceSetConfig("extra", "eyes"),
        PieceSetConfig("extra", "falcon"),
        PieceSetConfig("extra", "fantasy_alt"),
        PieceSetConfig("extra", "game_room"),
        PieceSetConfig("extra", "glass"),
        PieceSetConfig("extra", "gothic"),
        PieceSetConfig("extra", "graffiti"),
        PieceSetConfig("extra", "icy_sea"),
        PieceSetConfig("extra", "iowa"),
        PieceSetConfig("extra", "light"),
        PieceSetConfig("extra", "lolz"),
        PieceSetConfig("extra", "marble"),
        PieceSetConfig("extra", "maya"),
        PieceSetConfig("extra", "metal"),
        PieceSetConfig("extra", "modern"),
        PieceSetConfig("extra", "nature"),
        PieceSetConfig("extra", "neo"),
        PieceSetConfig("extra", "neo_wood"),
        PieceSetConfig("extra", "neon"),
        PieceSetConfig("extra", "newspaper"),
        PieceSetConfig("extra", "ocean"),
        PieceSetConfig("extra", "oslo"),
        PieceSetConfig("extra", "royale"),
        PieceSetConfig("extra", "sky"),
        PieceSetConfig("extra", "space"),
        PieceSetConfig("extra", "spatial"),
        PieceSetConfig("extra", "tigers"),
        PieceSetConfig("extra", "tournament"),
        PieceSetConfig("extra", "vintage"),
        PieceSetConfig("extra", "wood"),
        PieceSetConfig("custom", "a"),
        PieceSetConfig("custom", "b"),
        PieceSetConfig("custom", "c"),
        PieceSetConfig("custom", "d"),
        PieceSetConfig("custom", "e"),
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


def list_board_theme_paths(game: str) -> list[Path]:
    game_dir = Path(f"resources/board_themes/{game}")
    if not game_dir.exists():
        return []
    return [
        p for p in game_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]


def open_board_theme(path: Path, board_w: int, board_h: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    img.putalpha(255)
    return img.resize((board_w, board_h))
