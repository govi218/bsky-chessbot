import random
import torch
from torch.utils.data import Dataset

from src import common
from src.pgn_parser import iter_pgn_games, parse_pgn_game, parse_variant_tag, replay_moves_to_fens


class ChessBoardOrientationDataset(Dataset):

    def __init__(self, pgn_file_name, rotate_probability=0.3, max=100000):
        self.board_list = []
        self.rotate_probability = rotate_probability

        for game_text in iter_pgn_games(pgn_file_name):
            parsed = parse_pgn_game(game_text)
            variant, chess960 = parse_variant_tag(parsed.tags.get("Variant", "chess"))
            if variant != "chess":
                continue

            initial_fen = parsed.tags.get("FEN")
            try:
                fens = replay_moves_to_fens(
                    parsed.moves,
                    variant=variant,
                    initial_fen=initial_fen,
                    chess960=chess960,
                )
            except Exception:
                continue

            for fen in fens:
                position = common.position_from_notation(fen, game="chess")
                if position is not None:
                    self.board_list.append(position)
                if len(self.board_list) >= max:
                    break
            if len(self.board_list) >= max:
                break

        random.shuffle(self.board_list)

        print(f"Found {len(self.board_list)} positions")

    def __len__(self):
        return len(self.board_list)

    def __getitem__(self, idx):
        rotate = random.uniform(0.0, 1.0) < self.rotate_probability

        input = common.chess_board_to_tensor(self.board_list[idx])
        if rotate:
            input = common.rotate_board_tensor(input)

        return (input, torch.tensor([1.0 if rotate else 0.0]))


def test_data_set():
    pgn_file = "resources/lichess_games/lichess_db_standard_rated_2013-05.pgn"

    c = ChessBoardOrientationDataset(pgn_file, max=100)

    for i in range(0, len(c)):
        input, target = c[i]

        assert not input.isnan().any()
        print(common.tensor_to_chess_board(input).fen())
        print("flipped:", target.item() == 1.0)
