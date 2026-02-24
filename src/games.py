from dataclasses import dataclass


@dataclass(frozen=True)
class GameSpec:
    key: str
    board_rows: int
    board_cols: int
    piece_symbols: tuple[str, ...]
    color_pairs: tuple[tuple[str, str], ...]

    @property
    def num_squares(self) -> int:
        return self.board_rows * self.board_cols

    @property
    def piece_set(self) -> set[str]:
        return set(self.piece_symbols)

    @property
    def color_swap_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for a, b in self.color_pairs:
            mapping[a] = b
            mapping[b] = a
        return mapping


CHESS = GameSpec(
    key="chess",
    board_rows=8,
    board_cols=8,
    piece_symbols=("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"),
    color_pairs=(
        ("P", "p"),
        ("N", "n"),
        ("B", "b"),
        ("R", "r"),
        ("Q", "q"),
        ("K", "k"),
    ),
)


XIANGQI = GameSpec(
    key="xiangqi",
    board_rows=10,
    board_cols=9,
    piece_symbols=("R", "N", "B", "A", "K", "C", "P", "r", "n", "b", "a", "k", "c", "p"),
    color_pairs=(
        ("R", "r"),
        ("N", "n"),
        ("B", "b"),
        ("A", "a"),
        ("K", "k"),
        ("C", "c"),
        ("P", "p"),
    ),
)


GAMES: dict[str, GameSpec] = {
    CHESS.key: CHESS,
    XIANGQI.key: XIANGQI,
}


def get_game(game: str | GameSpec) -> GameSpec:
    if isinstance(game, GameSpec):
        return game

    key = game.strip().lower()
    if key in {"chinese_chess", "chinese-chess", "xq"}:
        key = "xiangqi"

    if key not in GAMES:
        raise ValueError(f"Unsupported game: {game}. Supported games: {list(GAMES.keys())}")
    return GAMES[key]
