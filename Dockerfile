FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    stockfish \
    curl \
    libcairo2-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/usr/games:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY chessbot ./chessbot
COPY chess_diagram_to_fen.py .
COPY main.py .
COPY src ./src
COPY resources ./resources
COPY models ./models

RUN uv sync --extra cpu

RUN if [ ! -f "models/chess/best_model_position_0.977_2026-03-09-00-47-01.pth" ]; then \
        ./download_models.sh; \
    fi

ENV BLUESKY_PASSWORD=""
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH="/usr/lib:$LD_LIBRARY_PATH"

CMD ["uv", "run", "python", "-m", "chessbot.listener"]
