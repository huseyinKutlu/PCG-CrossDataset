#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$HOME/Desktop/pcg_project"
RAW_DIR="$PROJECT_ROOT/data/raw"
CINC_DIR="$RAW_DIR/cinc2016"
CIRCOR_DIR="$RAW_DIR/circor2022"
TMP_DIR="$PROJECT_ROOT/data/_tmp"

rm -rf "$CINC_DIR" "$CIRCOR_DIR" "$TMP_DIR"
mkdir -p "$CINC_DIR" "$CIRCOR_DIR" "$TMP_DIR"

echo ">>> Disk durumu:"
df -h "$PROJECT_ROOT" | tail -1

# CinC 2016
echo ""
echo ">>> [1/2] CinC 2016 ZIP indiriliyor (~170 MB)..."
cd "$TMP_DIR"
wget -c --show-progress -O cinc.zip \
  "https://physionet.org/static/published-projects/challenge-2016/classification-of-heart-sound-recordings-the-physionetcomputing-in-cardiology-challenge-2016-1.0.0.zip"

echo ">>> CinC aciliyor..."
unzip -q -o cinc.zip -d cinc_unz
INNER=$(find cinc_unz -maxdepth 2 -type d -name "training-a" | head -1)
if [ -z "$INNER" ]; then
    echo "HATA: CinC ZIP icinde training-a yok"
    find cinc_unz -maxdepth 3 -type d
    exit 1
fi
PARENT=$(dirname "$INNER")
mv "$PARENT"/* "$CINC_DIR/"
rm -rf cinc_unz cinc.zip

echo ">>> CinC kontrol:"
echo "  WAV: $(find "$CINC_DIR" -name '*.wav' | wc -l)"
echo "  REFERENCE.csv: $(find "$CINC_DIR" -name 'REFERENCE.csv' | wc -l)"
find "$CINC_DIR" -maxdepth 1 -type d -name "training-*" | sort

# CirCor 2022
echo ""
echo ">>> [2/2] CirCor 2022 ZIP indiriliyor (~1.6 GB)..."
cd "$TMP_DIR"
wget -c --show-progress -O circor.zip \
  "https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip"

echo ">>> CirCor aciliyor..."
unzip -q -o circor.zip -d circor_unz
INNER_CC=$(find circor_unz -maxdepth 2 -type d -name "training_data" | head -1)
if [ -z "$INNER_CC" ]; then
    echo "HATA: CirCor ZIP icinde training_data yok"
    find circor_unz -maxdepth 3 -type d
    exit 1
fi
PARENT_CC=$(dirname "$INNER_CC")
mv "$PARENT_CC"/* "$CIRCOR_DIR/"
rm -rf circor_unz circor.zip
rmdir "$TMP_DIR" 2>/dev/null

echo ">>> CirCor kontrol:"
echo "  WAV: $(find "$CIRCOR_DIR" -name '*.wav' | wc -l)"
[ -f "$CIRCOR_DIR/training_data.csv" ] && echo "  training_data.csv VAR" || echo "  training_data.csv YOK"

echo ""
echo "=========================================="
echo ">>> TAMAMLANDI"
echo "=========================================="
du -sh "$RAW_DIR"
