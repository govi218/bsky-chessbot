import argparse
import importlib
import inspect
import os


DATASET = "dataset"
TRAIN = "train"
EVAL = "eval"

BBOX = "bbox"
POSITION = "position"
ORIENTATION = "orientation"
IMAGE_ROTATION = "image_rotation"
EXISTENCE = "existence"


FUNCTION_TARGETS = {
    (DATASET, BBOX): ("src.bounding_box.dataset", "test_data_set"),
    (TRAIN, BBOX): ("src.bounding_box.train", "train"),

    (DATASET, POSITION): ("src.fen_recognition.dataset", "test_data_set"),
    (TRAIN, POSITION): ("src.fen_recognition.train", "train"),
    (EVAL, POSITION): ("src.fen_recognition.show_wrong_evals", "show_wrong_fens"),

    (DATASET, ORIENTATION): ("src.board_orientation.dataset", "test_data_set"),
    (TRAIN, ORIENTATION): ("src.board_orientation.train", "train"),
    (EVAL, ORIENTATION): ("src.board_orientation.show_wrong_evals", "show_wrong_orientation_evals"),

    (DATASET, IMAGE_ROTATION): ("src.board_image_rotation.dataset", "test_data_set"),
    (TRAIN, IMAGE_ROTATION): ("src.board_image_rotation.train", "train"),
    (EVAL, IMAGE_ROTATION): ("src.board_image_rotation.show_wrong_evals", "show_wrong_image_rotations"),

    (DATASET, EXISTENCE): ("src.existence.dataset", "test_data_set"),
    (TRAIN, EXISTENCE): ("src.existence.train", "train"),
    (EVAL, EXISTENCE): ("src.existence.show_wrong_evals", "show_wrong_existence"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training/data entrypoint")
    parser.add_argument("function", choices=[DATASET, TRAIN, EVAL], help="the function to run")
    parser.add_argument(
        "model",
        nargs="?",
        choices=[BBOX, POSITION, ORIENTATION, IMAGE_ROTATION, EXISTENCE, None],
        default=None,
        help="the model/pipeline to use",
    )
    parser.add_argument("--dir", type=str, help="directory that contains diagram images")
    parser.add_argument("--game", type=str, required=True, help="game key from src/games.py (e.g. chess, xiangqi, shogi)")
    parser.add_argument("--pgn", type=str, help="single PGN file path")
    parser.add_argument("--pgn_train", type=str, help="train PGN file path")
    parser.add_argument("--pgn_test", type=str, help="test PGN file path")
    parser.add_argument("--model_path", type=str, help="model checkpoint path for eval commands")
    args = parser.parse_args()

    # In headless environments, force a non-GUI matplotlib backend for training.
    # This avoids Tk backend crashes when generating plots at the end of training.
    if (
        args.function == TRAIN
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ.setdefault("MPLBACKEND", "Agg")

    selection = (args.function, args.model)
    if selection not in FUNCTION_TARGETS:
        raise Exception(f"Selection {selection} not supported\nSupported selections: {list(FUNCTION_TARGETS.keys())}")

    module_name, fn_name = FUNCTION_TARGETS[selection]
    fn = getattr(importlib.import_module(module_name), fn_name)
    sig = inspect.signature(fn)

    kwargs = {}
    if "game" in sig.parameters:
        kwargs["game"] = args.game
    if "root_dir" in sig.parameters and args.dir is not None:
        kwargs["root_dir"] = args.dir
    if "data_root_dir" in sig.parameters and args.dir is not None:
        kwargs["data_root_dir"] = args.dir
    if "pgn_file" in sig.parameters and args.pgn is not None:
        kwargs["pgn_file"] = args.pgn
    if "pgn_file_name" in sig.parameters and args.pgn is not None:
        kwargs["pgn_file_name"] = args.pgn
    if "pgn_dir" in sig.parameters and args.pgn is not None:
        kwargs["pgn_dir"] = args.pgn
    if "train_pgn_file" in sig.parameters and args.pgn_train is not None:
        kwargs["train_pgn_file"] = args.pgn_train
    if "test_pgn_file" in sig.parameters and args.pgn_test is not None:
        kwargs["test_pgn_file"] = args.pgn_test
    if "model_path" in sig.parameters and args.model_path is not None:
        kwargs["model_path"] = args.model_path
    if "model_file" in sig.parameters and args.model_path is not None:
        kwargs["model_file"] = args.model_path

    fn(**kwargs)
