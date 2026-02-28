from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from cairosvg import svg2png
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
        PieceSetConfig("pychess", "atopdown"),
        PieceSetConfig("pychess", "luffy"),
        PieceSetConfig("pychess", "santa"),
        PieceSetConfig("other", "chess_1Kbyte_gambit"),
        PieceSetConfig("other", "chess_kaneo"),
        PieceSetConfig("other", "chess_kaneo_midnight"),
        PieceSetConfig("other", "chess_maestro_bw"),
    ),
    piece_file_names_by_provider={
        "lichess": ("bB.svg", "bK.svg", "bN.svg", "bP.svg", "bQ.svg", "bR.svg", "wB.svg", "wK.svg", "wN.svg", "wP.svg", "wQ.svg", "wR.svg"),
        "extra": ("bb.png", "bk.png", "bn.png", "bp.png", "bq.png", "br.png", "wb.png", "wk.png", "wn.png", "wp.png", "wq.png", "wr.png"),
        "custom": ("bb.png", "bk.png", "bn.png", "bp.png", "bq.png", "br.png", "wb.png", "wk.png", "wn.png", "wp.png", "wq.png", "wr.png"),
        "pychess": ("bB.svg", "bK.svg", "bN.svg", "bP.svg", "bQ.svg", "bR.svg", "wB.svg", "wK.svg", "wN.svg", "wP.svg", "wQ.svg", "wR.svg",
                    "bb.svg", "bk.svg", "bn.svg", "bp.svg", "bq.svg", "br.svg", "wb.svg", "wk.svg", "wn.svg", "wp.svg", "wq.svg", "wr.svg",
                    "bB.png", "bK.png", "bN.png", "bP.png", "bQ.png", "bR.png", "wB.png", "wK.png", "wN.png", "wP.png", "wQ.png", "wR.png"),
        "other": ("bB.svg", "bK.svg", "bN.svg", "bP.svg", "bQ.svg", "bR.svg", "wB.svg", "wK.svg", "wN.svg", "wP.svg", "wQ.svg", "wR.svg"),
    },
)


XIANGQI_RENDER_CONFIG = GameRenderConfig(
    piece_sets=(
        PieceSetConfig("pychess", "2dhanzi"),
        PieceSetConfig("pychess", "2dintl"),
        PieceSetConfig("pychess", "ct2"),
        PieceSetConfig("pychess", "ct2w"),
        PieceSetConfig("pychess", "ct3"),
        PieceSetConfig("pychess", "euro"),
        PieceSetConfig("pychess", "eventhanzi"),
        PieceSetConfig("pychess", "eventintl"),
        PieceSetConfig("pychess", "hnz"),
        PieceSetConfig("pychess", "hnzw"),
        PieceSetConfig("pychess", "Ka"),
        PieceSetConfig("pychess", "lishu"),
        PieceSetConfig("pychess", "lishuw"),
        PieceSetConfig("pychess", "playok"),
        PieceSetConfig("pychess", "ttxqhanzi"),
        PieceSetConfig("pychess", "ttxqintl"),
        PieceSetConfig("pychess", "wikim"),
        PieceSetConfig("other", "xiangqi_gmchess_alternative_black_red"),
        PieceSetConfig("other", "xiangqi_gmchess_alternative_gray"),
        PieceSetConfig("other", "xiangqi_gmchess_style_wood"),
        PieceSetConfig("other", "xiangqi_wikipedia_intl_modded"),
        PieceSetConfig("other", "ccbridge_3_0_beta4_default_preview_remake"),
        PieceSetConfig("other", "commons_xiangqi_pieces_print_2010"),
        PieceSetConfig("other", "commons_xiangqi_pieces_print_2010_bw_heavy"),
        PieceSetConfig("other", "euro_xiangqi_js"),
        PieceSetConfig("other", "euro_xiangqi_js_tricolor"),
        PieceSetConfig("other", "latex_xqlarge_2006_chinese_autotrace"),
        PieceSetConfig("other", "latex_xqlarge_2006_chinese_potrace"),
        PieceSetConfig("other", "playok_2014_chinese"),
        PieceSetConfig("other", "playok_2014_chinese_noshadow"),
        PieceSetConfig("other", "retro_simple"),
    ),
    piece_file_names_by_provider={
        "pychess": (
            "wR.svg", "wN.svg", "wB.svg", "wA.svg", "wK.svg", "wC.svg", "wP.svg",
            "bR.svg", "bN.svg", "bB.svg", "bA.svg", "bK.svg", "bC.svg", "bP.svg",
        ),
        "other": (
            "wR.svg", "wN.svg", "wB.svg", "wA.svg", "wK.svg", "wC.svg", "wP.svg",
            "bR.svg", "bN.svg", "bB.svg", "bA.svg", "bK.svg", "bC.svg", "bP.svg",
        ),
    },
)


SHOGI_RENDER_CONFIG = GameRenderConfig(
    piece_sets=(
        PieceSetConfig("pychess", "2kanji"),
        PieceSetConfig("pychess", "bnw"),
        PieceSetConfig("pychess", "bw"),
        PieceSetConfig("pychess", "ctim"),
        PieceSetConfig("pychess", "ctk"),
        PieceSetConfig("pychess", "ctkw3d"),
        PieceSetConfig("pychess", "ctm"),
        PieceSetConfig("pychess", "ctp"),
        PieceSetConfig("pychess", "ctp3d"),
        PieceSetConfig("pychess", "ctw"),
        PieceSetConfig("pychess", "cz"),
        PieceSetConfig("pychess", "Ka"),
        PieceSetConfig("pychess", "Portella-Intl"),
        PieceSetConfig("pychess", "Portella-Kanji"),
        PieceSetConfig("other", "2-kanji_red_wood"),
        PieceSetConfig("other", "international"),
        PieceSetConfig("other", "kanji_brown"),
        PieceSetConfig("other", "kanji_light"),
        PieceSetConfig("other", "kanji_light_3D_OTB"),
        PieceSetConfig("other", "kanji_red_wood"),
        PieceSetConfig("other", "military"),
        PieceSetConfig("lishogi", "1kanji_3d"),
        PieceSetConfig("lishogi", "2kanji_3d"),
        PieceSetConfig("lishogi", "alfaerie"),
        PieceSetConfig("lishogi", "better_8_bit"),
        PieceSetConfig("lishogi", "characters"),
        PieceSetConfig("lishogi", "dewitt_1kanji"),
        PieceSetConfig("lishogi", "dewitt_2kanji"),
        PieceSetConfig("lishogi", "dewitt_czech"),
        PieceSetConfig("lishogi", "dobutsu"),
        PieceSetConfig("lishogi", "engraved_cz"),
        PieceSetConfig("lishogi", "engraved_cz_bnw"),
        PieceSetConfig("lishogi", "firi"),
        PieceSetConfig("lishogi", "glass"),
        PieceSetConfig("lishogi", "greenwade"),
        PieceSetConfig("lishogi", "hitomoji"),
        PieceSetConfig("lishogi", "international"),
        PieceSetConfig("lishogi", "intl_colored_2d"),
        PieceSetConfig("lishogi", "intl_colored_3d"),
        PieceSetConfig("lishogi", "intl_monochrome_2d"),
        PieceSetConfig("lishogi", "intl_portella"),
        PieceSetConfig("lishogi", "intl_shadowed"),
        PieceSetConfig("lishogi", "intl_wooden_3d"),
        PieceSetConfig("lishogi", "joyful"),
        PieceSetConfig("lishogi", "kanji_brown"),
        PieceSetConfig("lishogi", "kanji_guide_shadowed"),
        PieceSetConfig("lishogi", "kanji_light"),
        PieceSetConfig("lishogi", "kanji_red_wood"),
        PieceSetConfig("lishogi", "logy_games"),
        PieceSetConfig("lishogi", "mnemonic"),
        PieceSetConfig("lishogi", "orangain"),
        PieceSetConfig("lishogi", "pixel"),
        PieceSetConfig("lishogi", "portella"),
        PieceSetConfig("lishogi", "portella_2kanji"),
        PieceSetConfig("lishogi", "ryoko_1kanji"),
        PieceSetConfig("lishogi", "shogi_bnw"),
        PieceSetConfig("lishogi", "shogi_cz"),
        PieceSetConfig("lishogi", "shogi_fcz"),
        PieceSetConfig("lishogi", "simple_kanji"),
        PieceSetConfig("lishogi", "vald_opt"),
        PieceSetConfig("lishogi", "valdivia"),
        PieceSetConfig("lishogi", "western"),
        PieceSetConfig("shogi-themes", "13xforever_1kanji"),
        PieceSetConfig("shogi-themes", "13xforever_2kanji"),
        PieceSetConfig("shogi-themes", "hari_seldon_1kanji"),
        PieceSetConfig("shogi-themes", "hari_seldon_2kanji"),
        PieceSetConfig("shogi-themes", "kinki_1kanji"),
        PieceSetConfig("shogi-themes", "kinki_2kanji"),
        PieceSetConfig("shogi-themes", "minase_1kanji"),
        PieceSetConfig("shogi-themes", "minase_2kanji"),
        PieceSetConfig("shogi-themes", "ryoko_1kanji"),
        PieceSetConfig("shogi-themes", "ryoko_2kanji"),
    ),
    piece_file_names_by_provider={
        "pychess": (
            "wK.svg", "wR.svg", "wB.svg", "wG.svg", "wS.svg", "wN.svg", "wL.svg", "wP.svg",
            "w+R.svg", "w+B.svg", "w+S.svg", "w+N.svg", "w+L.svg", "w+P.svg",
            "bK.svg", "bR.svg", "bB.svg", "bG.svg", "bS.svg", "bN.svg", "bL.svg", "bP.svg",
            "b+R.svg", "b+B.svg", "b+S.svg", "b+N.svg", "b+L.svg", "b+P.svg",
            "wK.png", "wR.png", "wB.png", "wG.png", "wS.png", "wN.png", "wL.png", "wP.png",
            "w+R.png", "w+B.png", "w+S.png", "w+N.png", "w+L.png", "w+P.png",
            "bK.png", "bR.png", "bB.png", "bG.png", "bS.png", "bN.png", "bL.png", "bP.png",
            "b+R.png", "b+B.png", "b+S.png", "b+N.png", "b+L.png", "b+P.png",
        ),
        "other": (
            "wK.svg", "wR.svg", "wB.svg", "wG.svg", "wS.svg", "wN.svg", "wL.svg", "wP.svg",
            "w+R.svg", "w+B.svg", "w+S.svg", "w+N.svg", "w+L.svg", "w+P.svg",
            "bK.svg", "bR.svg", "bB.svg", "bG.svg", "bS.svg", "bN.svg", "bL.svg", "bP.svg",
            "b+R.svg", "b+B.svg", "b+S.svg", "b+N.svg", "b+L.svg", "b+P.svg",
        ),
        "lishogi": (
            "wK.svg", "wR.svg", "wB.svg", "wG.svg", "wS.svg", "wN.svg", "wL.svg", "wP.svg",
            "w+R.svg", "w+B.svg", "w+S.svg", "w+N.svg", "w+L.svg", "w+P.svg",
            "bK.svg", "bR.svg", "bB.svg", "bG.svg", "bS.svg", "bN.svg", "bL.svg", "bP.svg",
            "b+R.svg", "b+B.svg", "b+S.svg", "b+N.svg", "b+L.svg", "b+P.svg",
            "wK.png", "wR.png", "wB.png", "wG.png", "wS.png", "wN.png", "wL.png", "wP.png",
            "w+R.png", "w+B.png", "w+S.png", "w+N.png", "w+L.png", "w+P.png",
            "bK.png", "bR.png", "bB.png", "bG.png", "bS.png", "bN.png", "bL.png", "bP.png",
            "b+R.png", "b+B.png", "b+S.png", "b+N.png", "b+L.png", "b+P.png",
        ),
        "shogi-themes": (
            "wK.svg", "wR.svg", "wB.svg", "wG.svg", "wS.svg", "wN.svg", "wL.svg", "wP.svg",
            "w+R.svg", "w+B.svg", "w+S.svg", "w+N.svg", "w+L.svg", "w+P.svg",
            "bK.svg", "bR.svg", "bB.svg", "bG.svg", "bS.svg", "bN.svg", "bL.svg", "bP.svg",
            "b+R.svg", "b+B.svg", "b+S.svg", "b+N.svg", "b+L.svg", "b+P.svg",
        ),
    },
)


PLACEHOLDER_RENDER_CONFIG = GameRenderConfig(
    piece_sets=(PieceSetConfig("placeholder", "default", use_placeholders=True),),
    piece_file_names_by_provider={},
)


GAME_RENDER_CONFIGS: dict[str, GameRenderConfig] = {
    "chess": CHESS_RENDER_CONFIG,
    "xiangqi": XIANGQI_RENDER_CONFIG,
    "shogi": SHOGI_RENDER_CONFIG,
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
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
    ]


def open_board_theme(path: Path, board_w: int, board_h: int) -> Image.Image:
    if path.suffix.lower() == ".svg":
        with open(path, "rb") as f:
            svg_data = f.read()
        png_data = svg2png(
            bytestring=svg_data, output_width=board_w
        )
        img = Image.open(BytesIO(png_data)).convert("RGBA")
    else:
        img = Image.open(path).convert("RGBA")
    img.putalpha(255)
    return img.resize((board_w, board_h))
