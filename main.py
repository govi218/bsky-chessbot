import argparse
import inspect

import src.board_image_rotation.dataset as image_rotation_data
import src.board_image_rotation.show_wrong_evals as image_rotation_eval
import src.board_image_rotation.train as image_rotation_train
import src.board_orientation.dataset as orientation_data
import src.board_orientation.show_wrong_evals as orientation_eval
import src.board_orientation.train as orientation_train
import src.bounding_box.dataset as bbox_data
import src.bounding_box.generate_chessboards_bbox as bbox_gen
import src.bounding_box.train as bbox_train
import src.existence.dataset as existence_data
import src.existence.generate_existence as existence_gen
import src.existence.show_wrong_evals as existence_eval
import src.existence.train as existence_train
import src.fen_recognition.dataset as position_data
import src.fen_recognition.generate_chessboards as position_gen
import src.fen_recognition.show_wrong_evals as position_eval
import src.fen_recognition.train as position_train


GENERATE = "generate"
DATASET = "dataset"
TRAIN = "train"
EVAL = "eval"

BBOX = "bbox"
POSITION = "position"
FEN = "fen"  # backward-compatible alias
ORIENTATION = "orientation"
IMAGE_ROTATION = "image_rotation"
EXISTENCE = "existence"


functions = {
    (GENERATE, BBOX): bbox_gen.generate_bbox_training_data,
    (DATASET, BBOX): bbox_data.test_data_set,
    (TRAIN, BBOX): bbox_train.train,

    (GENERATE, POSITION): position_gen.generate_fen_training_data,
    (DATASET, POSITION): position_data.test_data_set,
    (TRAIN, POSITION): position_train.train,
    (EVAL, POSITION): position_eval.show_wrong_fens,

    (GENERATE, FEN): position_gen.generate_fen_training_data,
    (DATASET, FEN): position_data.test_data_set,
    (TRAIN, FEN): position_train.train,
    (EVAL, FEN): position_eval.show_wrong_fens,

    (DATASET, ORIENTATION): orientation_data.test_data_set,
    (TRAIN, ORIENTATION): orientation_train.train,
    (EVAL, ORIENTATION): orientation_eval.show_wrong_orientation_evals,

    (DATASET, IMAGE_ROTATION): image_rotation_data.test_data_set,
    (TRAIN, IMAGE_ROTATION): image_rotation_train.train,
    (EVAL, IMAGE_ROTATION): image_rotation_eval.show_wrong_image_rotations,

    (GENERATE, EXISTENCE): existence_gen.generate_existence_training_data,
    (DATASET, EXISTENCE): existence_data.test_data_set,
    (TRAIN, EXISTENCE): existence_train.train,
    (EVAL, EXISTENCE): existence_eval.show_wrong_existence,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training/data entrypoint")
    parser.add_argument("function", choices=[GENERATE, DATASET, TRAIN, EVAL], help="the function to run")
    parser.add_argument(
        "model",
        nargs="?",
        choices=[BBOX, POSITION, FEN, ORIENTATION, IMAGE_ROTATION, EXISTENCE, None],
        default=None,
        help="the model/pipeline to use",
    )
    parser.add_argument("--dir", type=str, help="directory that contains diagram images")
    parser.add_argument("--game", type=str, default="chess", help="game key from src/games.py (e.g. chess, xiangqi)")
    args = parser.parse_args()

    selection = (args.function, args.model)
    if selection not in functions:
        raise Exception(f"Selection {selection} not supported\nSupported selections: {list(functions.keys())}")

    fn = functions[selection]
    sig = inspect.signature(fn)

    kwargs = {}
    if "game" in sig.parameters:
        kwargs["game"] = args.game
    if "root_dir" in sig.parameters and args.dir is not None:
        kwargs["root_dir"] = args.dir
    if "data_root_dir" in sig.parameters and args.dir is not None:
        kwargs["data_root_dir"] = args.dir

    fn(**kwargs)
