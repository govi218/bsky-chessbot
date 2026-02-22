import random
import torch

from src import common
from src.board_orientation.model import OrientationModel
from src.pgn_parser import iter_pgn_games, parse_pgn_game, parse_variant_tag, replay_moves_to_fens


@torch.no_grad()
def show_wrong_orientation_evals(
    rotate_probability=0.5,
    no_rotate_bias=0.0,
    pgn_file="resources/lichess_games/lichess_db_standard_rated_2013-05.pgn",
    model_file="models/best_model_orientation_0.987_2024-02-04-17-34-05.pth",
):
    model = OrientationModel()
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    model.eval()

    num_samples = 0
    num_correct = 0

    for game_text in iter_pgn_games(pgn_file):
        parsed = parse_pgn_game(game_text)
        variant, chess960 = parse_variant_tag(parsed.tags.get("Variant", "chess"))
        if variant != "chess":
            continue

        initial_fen = parsed.tags.get("FEN")
        try:
            fens = replay_moves_to_fens(parsed.moves, variant=variant, initial_fen=initial_fen, chess960=chess960)
        except Exception:
            continue

        for fen in fens:
            position = common.position_from_notation(fen, game="chess")
            if position is None:
                continue

            input_tensor = common.chess_board_to_tensor(position)
            rotate = random.uniform(0.0, 1.0) < rotate_probability
            if rotate:
                input_tensor = common.rotate_board_tensor(input_tensor)
            target = 1.0 if rotate else 0.0

            output = model(input_tensor.unsqueeze(0)).squeeze(0)
            output -= no_rotate_bias

            print(common.tensor_to_chess_board(input_tensor).fen())

            num_samples += 1
            if abs(output.item() - target) < 0.5:
                num_correct += 1
            else:
                print("Wrong: ", (output.item(), target))
                print(num_correct / num_samples)
                input("Press enter to continue ...")
