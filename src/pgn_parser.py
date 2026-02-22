import re
from dataclasses import dataclass
from pathlib import Path


try:
    import pyffish as sf
except ImportError:
    sf = None


@dataclass(frozen=True)
class ParsedPGN:
    tags: dict[str, str]
    moves: list[str]


RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}


def _require_pyffish():
    if sf is None:
        raise RuntimeError(
            "pyffish is required for PGN replay but is not installed. Install project dependencies including pyffish."
        )


def parse_variant_tag(raw_variant: str | None) -> tuple[str, bool]:
    raw = (raw_variant or "chess").strip().lower()
    chess960 = "960" in raw or "random" in raw
    variant = raw.removesuffix("960")
    if variant == "caparandom":
        return "capablanca", True
    if variant == "fischerandom":
        return "chess", True
    return variant or "chess", chess960


def parse_pgn_tags(pgn_text: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    regex = re.compile(r'^\s*\[([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\]\s*$', re.MULTILINE)
    for match in regex.finditer(pgn_text):
        tags[match.group(1)] = match.group(2).replace('\\"', '"')
    return tags


def _strip_pgn_noise(move_text: str) -> str:
    move_text = re.sub(r"\{[^}]*\}", " ", move_text)
    move_text = re.sub(r";[^\n]*", " ", move_text)
    move_text = re.sub(r"\([^)]*\)", " ", move_text)
    move_text = re.sub(r"\$\d+", " ", move_text)
    return move_text


def extract_mainline_moves(move_text: str) -> list[str]:
    text = _strip_pgn_noise(move_text)
    tokens = re.split(r"\s+", text.strip())
    moves: list[str] = []
    for tok in tokens:
        if not tok:
            continue
        if tok in RESULT_TOKENS:
            continue
        if re.match(r"^\d+\.(\.\.)?$", tok):
            continue
        tok = re.sub(r"^\d+\.(\.\.)?", "", tok)
        if not tok or tok in RESULT_TOKENS:
            continue
        moves.append(tok)
    return moves


def parse_pgn_game(pgn_text: str) -> ParsedPGN:
    tags = parse_pgn_tags(pgn_text)
    body = re.sub(r'^\s*\[[^\]]*\]\s*$', "", pgn_text, flags=re.MULTILINE)
    moves = extract_mainline_moves(body)
    return ParsedPGN(tags=tags, moves=moves)


def replay_moves_to_fens(
    moves: list[str], variant: str = "chess", initial_fen: str | None = None, chess960: bool = False
) -> list[str]:
    _require_pyffish()
    fen = initial_fen or sf.start_fen(variant)
    fens: list[str] = []
    for move in moves:
        try:
            fen = sf.get_fen(variant, fen, [move], chess960, False, False, 0)
        except TypeError:
            fen = sf.get_fen(variant, fen, [move], chess960)
        fens.append(fen)
    return fens


def iter_pgn_games(file_path: str | Path):
    current_lines: list[str] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("[Event ") and current_lines:
                game_text = "".join(current_lines).strip()
                if game_text:
                    yield game_text
                current_lines = [line]
                continue
            current_lines.append(line)
    if current_lines:
        game_text = "".join(current_lines).strip()
        if game_text:
            yield game_text
