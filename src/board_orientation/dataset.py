import random
import torch
from torch.utils.data import Dataset

from src import common
from src.games import get_game
from src.pgn_parser import iter_pgn_games, parse_pgn_game, parse_pgn_tags, parse_variant_tag, replay_moves_to_fens


class BoardOrientationDataset(Dataset):

    def __init__(self, pgn_file_name, game: str, rotate_probability=0.3, max=100000):
        self.game = get_game(game)
        self.board_list = []
        self.rotate_probability = rotate_probability

        for game_text in iter_pgn_games(pgn_file_name):
            tags = parse_pgn_tags(game_text)
            variant, chess960 = parse_variant_tag(tags.get("Variant", self.game.key))
            if variant != self.game.key:
                continue

            parsed = parse_pgn_game(game_text)

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
                position = common.position_from_notation(fen, game=self.game)
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

        input = common.position_to_tensor(self.board_list[idx])
        if rotate:
            input = common.rotate_tensor_180(input, self.game)

        return (input, torch.tensor([1.0 if rotate else 0.0]))


def test_data_set(game: str, pgn_file: str):
    spec = get_game(game)
    c = BoardOrientationDataset(pgn_file, game=spec, max=10000)
    print("len(c):", len(c))
    for i in range(0, 100):#len(c)):
        input, target = c[i]

        assert not input.isnan().any()
        print(common.tensor_to_position(input, game=spec).fen())
        print("flipped:", target.item() == 1.0)
