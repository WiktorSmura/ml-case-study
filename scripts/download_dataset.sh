#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Target directory
TARGET_DIR="data/raw"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Downloading 1999 Czech Financial Dataset into $TARGET_DIR..."

mkdir -p "$TARGET_DIR"

TMP_DIR="$(mktemp -d)"

uv run kaggle datasets download \
  -d mariammariamr/1999-czech-financial-dataset \
  --path "$TMP_DIR" \
  --unzip

INNER_DIR="$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)"

if [ -n "$INNER_DIR" ]; then
  cp -a "$INNER_DIR"/. "$TARGET_DIR"/
else
  cp -a "$TMP_DIR"/. "$TARGET_DIR"/
fi

rm -rf "$TMP_DIR"
