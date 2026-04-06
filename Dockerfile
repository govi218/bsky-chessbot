FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    stockfish \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY chessbot ./chessbot
COPY chess_diagram_to_fen.py .
COPY main.py .
COPY src ./src
COPY models ./models

RUN uv sync --extra cpu

RUN if [ ! -f "models/chess/best_model_position_0.977_2026-03-09-00-47-01.pth" ]; then \
        ./download_models.sh; \
    fi

ENV BLUESKY_PASSWORD=""
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "-m", "chessbot.listener"]
