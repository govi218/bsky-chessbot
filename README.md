# bsky-chessbot

Extract the FEN out of images of chess, xiangqi, or shogi diagrams. Includes Stockfish analysis for chess position evaluation and a Bluesky bot for automatic position analysis.

## Quick Start

```bash
git clone https://github.com/govi218/Chess_diagram_to_FEN.git
cd Chess_diagram_to_FEN
uv sync --extra cpu
./download_models.sh
brew install stockfish  # macOS
```

```python
from chessbot import analyze, format_result

result = analyze("screenshot.png")
print(format_result(result))
# FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
# White to move: +0.4, best: e4, line: 1. e4 e5 2. Nf3 Nc6
# Black to move: +0.4, best: e5, line: 1... e5 2. Nf3 Nc6
```

## Bluesky Bot

Mention `@chess.glados.computer` with an image (or reply to a post with an image) to get automatic position analysis.

```bash
# Set up environment
echo "BLUESKY_PASSWORD=your_password" > .env

# Run the bot
uv run python -m chessbot.listener
```

The bot will:
- Detect chess positions in images
- Show evaluation and best moves for both white and black to move
- Include "white" or "black" in your mention to specify turn
- Handle reply comments (analyzes parent post's image)

## Docker

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t chessbot .
docker run -d --env-file .env chessbot
```

## How it works

1. Detect if there exists any chess board in the image
2. Get a bounding box of the (most prominent) chess board
3. Check if the board image is rotated by 0, 90, 180, or 270 degrees
4. Finally detect the FEN by looking at each square tile and predicting the piece
5. Detect if the perspective is from blacks or whites perspective (using a simple fully connected NN)

All these steps (except the 5th) basically use some common pretrained convolutional models available via torchvision with slightly modified heads. Detection is made robust using demanding generated training data and augmentations.

Chess works best, xiangqi works fine too, shogi doesn't work very well (and this program also doesn't handle pieces in hand).

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and then clone the project:

```shell
git clone "https://github.com/govi218/Chess_diagram_to_FEN.git"
cd Chess_diagram_to_FEN
```

Then install dependencies with the PyTorch variant that matches your hardware:

```shell
uv sync --extra cpu     # CPU (works everywhere, no GPU required)
uv sync --extra cuda    # CUDA 12.8  (NVIDIA GPUs)
uv sync --extra rocm    # ROCm 6.4   (AMD GPUs, Linux only)
```

Download pretrained models:

```shell
./download_models.sh
```

Install Stockfish for position analysis:

```shell
brew install stockfish  # macOS
sudo apt install stockfish  # Linux
```

## Usage

### Chessbot API (recommended)

Full pipeline with Stockfish analysis:

```python
from chessbot import analyze, format_result

result = analyze("screenshot.png", depth=20, turn="w")
print(format_result(result))
```

Individual modules:

```python
from chessbot import image_to_fen, analyze_fen

# Just FEN detection
fen = image_to_fen("screenshot.png")
print(fen.fen)

# Analyze existing FEN
analysis = analyze_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", depth=15)
print(f"Best move: {analysis.best_move_san}")
```

### Low-level API

```python
from PIL import Image
from chess_diagram_to_fen import get_fen

img = Image.open("your_image.jpg")
result = get_fen(
    img=img,
    game="chess",
    auto_rotate_image=True,
    auto_rotate_board=True
)

print(result.fen)
```

### CLI

```shell
uv run python chess_diagram_to_fen.py --game chess --dir resources/test_images/real_use_cases_chess/
```


## Train models yourself

```shell
bash ./train.sh # trains all chess, xiangqi and shogi models
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

```shell
uv run python main.py eval position --game chess --model_path models/chess/<position-model>.pth
uv run python main.py eval orientation --game chess --model_path models/chess/<orientation-model>.pth
uv run python main.py eval image_rotation --game chess --model_path models/chess/<image-rotation-model>.pth
uv run python main.py eval existence --game chess --model_path models/chess/<existence-model>.pth
```

## Examples

### Successes

<img src="./resources/examples/success/chess_Figure_5.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/chess_Figure_1.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/xiangqi_Figure_8.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/xiangqi_Figure_2.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/success/shogi_Figure_5.jpg" width="600px" style="border-radius: 20px;">


### Failures

<img src="./resources/examples/failure/chess_Figure_4.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/chess_Figure_7.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/xiangqi_Figure_5.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/xiangqi_Figure_7.jpg" width="600px" style="border-radius: 20px;">

<img src="./resources/examples/failure/shogi_Figure_1.jpg" width="600px" style="border-radius: 20px;">

There are more examples in [resources/examples](./resources/examples).
