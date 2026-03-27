#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/raw/sift"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
ARCHIVE="sift.tar.gz"

mkdir -p "$DATA_DIR"

echo "Downloading SIFT dataset..."
curl -L -o "$DATA_DIR/$ARCHIVE" "$URL"

echo "Extracting..."
tar -xzf "$DATA_DIR/$ARCHIVE" -C "$DATA_DIR" --strip-components=1

rm "$DATA_DIR/$ARCHIVE"

echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
