# Chess diagram to FEN

Extract the FEN out of images of chess diagrams.

It works in multiple steps:
1. Detect if there exists any chess board in the image
2. Get a bounding box of the (most prominent) chess board
3. Check if the board image is rotated by 0, 90, 180, or 270 degrees
4. Finally detect the FEN by looking at each square tile and predicting the piece (but also getting the entire board as additional input to make distinguishing between black and white pieces easier)
5. Detect if the perspective is from blacks or whites perspective (using a simple fully connected NN)

All these steps (except the fifth) basically use some common pretrained convolutional models available via torchvision with slightly modified heads. Detection is made robust using demanding generated training data and augmentations.

## Multi-game pipeline

The training/data pipeline is now game-aware for rectangular grids and single-piece-per-square games.

- `src/games.py` defines game specs (`chess`, `xiangqi`), including board dimensions and piece alphabets.
- `src/common.py` contains game-agnostic helpers for:
  - parsing/normalizing piece-placement notation on rectangular boards,
  - converting board grids to/from tensors,
  - 180° rotation and color-flip transforms by game spec.
- `src/fen_recognition/*` is now generic (`BoardRec`, `BoardPositionDataset`) and accepts `game`/`tile_size`.
- `src/board_orientation/*` now replays PGN via `pyffish` and accepts `game`.

Current end-user image inference (`get_fen`) remains chess-only for model-weight compatibility, but training/eval entrypoints accept `--game`.

Xiangqi notation normalization example:

```python
from src import common

notation = common.normalize_position_notation(
    "rheakaehr_9_1c5c1_p1p1p1p1p_9_9_P1P1P1P1P_1C5C1_9_RHEAKAEHR",
    game="xiangqi",
)
print(notation)  # rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w
```

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and then clone the project:

```shell
git clone "https://github.com/tsoj/Chess_diagram_to_FEN.git"
cd Chess_diagram_to_FEN
```

Then install dependencies with the PyTorch variant that matches your hardware:

```shell
uv sync --extra cpu     # CPU (works everywhere, no GPU required)
uv sync --extra cuda    # CUDA 12.8  (NVIDIA GPUs)
uv sync --extra rocm    # ROCm 6.4   (AMD GPUs, Linux only)
```

If you want to use this repository as a dependency inside another Python project, install it as editable:

```shell
# from the consuming project (you might need to adjust the path to Chess_diagram_to_FEN)
uv add --editable ../Chess_diagram_to_FEN
```

## Usage

```python
from PIL import Image
from chess_diagram_to_fen import get_fen

img = Image.open("your_image.jpg")
result = get_fen(
    img=img,
    num_tries=10,
    auto_rotate_image=True,
    auto_rotate_board=True
)

print(result.fen)
```

Or use the demo program:
```shell
uv run python chess_diagram_to_fen.py --dir resources/test_images/real_use_cases/
```


## Train models yourself

#### Generate training data
Needs about **40 GB** disk space.
```shell
uv run python main.py generate fen
uv run python main.py generate position --game chess

# It is important to generate the fen data before
# the bbox and existence data, since the bbox data generation
# relies on the fen training data

uv sync --extra train-data
./download_website_screenshots.sh
uv run python main.py generate bbox
uv run python main.py generate existence

./download_lichess_games.sh
```

Additionally you can download [this Kaggle dataset](https://www.kaggle.com/datasets/koryakinp/chess-positions), unzip it, and place it into `resources/fen_images` to further augment the training data for FEN detection.

#### Review datasets (optional)

```shell
uv run python main.py dataset bbox
uv run python main.py dataset fen
uv run python main.py dataset position --game chess
uv run python main.py dataset orientation
uv run python main.py dataset image_rotation
uv run python main.py dataset existence
```

#### Train

```shell
uv run python main.py train bbox
uv run python main.py train fen
uv run python main.py train position --game chess
uv run python main.py train orientation
uv run python main.py train image_rotation
uv run python main.py train existence
```

#### Evaluate (optional)

```shell
uv run python main.py eval fen
uv run python main.py eval position --game chess
uv run python main.py eval orientation
uv run python main.py eval image_rotation
```

## Examples

### Successes

<img src="./resources/examples/success/success_1.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/success_2.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/success_3.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/success_4.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/success_5.jpg" width="600px" style="border-radius: 20px;">


### Failures

<img src="./resources/examples/failure/failure_1.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/failure_2.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/failure_3.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/failure_4.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/failure_5.jpg" width="600px" style="border-radius: 20px;">
