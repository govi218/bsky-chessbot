# Chess diagram to FEN

Extract the FEN out of images of chess diagrams.

It works in multiple steps:
1. Detect if there exists any chess board in the image
2. Get a bounding box of the (most prominent) chess board
3. Check if the board image is rotated by 0, 90, 180, or 270 degrees
4. Finally detect the FEN by looking at each square tile and predicting the piece (but also getting the entire board as additional input to make distinguishing between black and white pieces easier)
5. Detect if the perspective is from blacks or whites perspective (using a simple fully connected NN)

All these steps (except the fifth) basically use some common pretrained convolutional models available via torchvision with slightly modified heads. Detection is made robust using demanding generated training data and augmentations.

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
    game="chess",
    num_tries=10,
    auto_rotate_image=True,
    auto_rotate_board=True
)

print(result.fen)
```

Or use the demo program:
```shell
uv run python chess_diagram_to_fen.py --game chess --dir resources/test_images/real_use_cases/
```


## Train models yourself

```shell
bash ./train.sh # trains chess, xiangqi and shogi models
```

... or alternatively manually go through the steps described below

#### Generate training data

```shell
./download_website_screenshots.sh
./download_pychess_games.sh
```

#### Review datasets (optional)

```shell
uv run python main.py dataset position --game chess
uv run python main.py dataset bbox --game chess
uv run python main.py dataset image_rotation --game chess
uv run python main.py dataset existence --game chess
uv run python main.py dataset orientation --game chess
```

#### Train

```shell
uv run python main.py train position --game chess
uv run python main.py train bbox --game chess
uv run python main.py train image_rotation --game chess
uv run python main.py train existence --game chess
uv run python main.py train orientation --game chess
```

#### Evaluate (optional)

TODO: test if this still works

```shell
uv run python main.py eval position --game chess --model_path models/<position-model>.pth
uv run python main.py eval orientation --game chess --pgn resources/lichess_games/lichess_db_standard_rated_2013-05.pgn --model_path models/<orientation-model>.pth
uv run python main.py eval image_rotation --game chess --model_path models/<image-rotation-model>.pth
uv run python main.py eval existence --game chess --model_path models/<existence-model>.pth
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
