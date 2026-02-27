#!/bin/bash

set -e

mkdir -p resources/pychess_games
cd resources/pychess_games

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

wget -O "$tmp_dir/repo.zip" https://github.com/gbtami/pychess-variants-games/archive/refs/heads/main.zip
unzip -o "$tmp_dir/repo.zip" -d "$tmp_dir"

find "$tmp_dir" -name "*.pgn.bz2" -exec mv {} . \;

for f in *.pgn.bz2; do
    [ -f "$f" ] || continue
    bzip2 -f -d "$f"
done
