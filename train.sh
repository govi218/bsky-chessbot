#!/bin/bash

set -e

[ -d models ] && mv models "models_backup_$(date +%Y%m%d_%H%M%S)"

uv sync --extra cuda # potentially use `--extra rocm` instead

[ ! -d resources/website_screenshots ] && ./download_website_screenshots.sh
[ ! -d resources/pychess_games ] && ./download_pychess_games.sh

games=(chess shogi xiangqi)
tasks=(orientation existence bbox image_rotation position)

for game in "${games[@]}"; do
  for task in "${tasks[@]}"; do
    uv run python -u main.py train "$task" --game "$game"
  done
done
