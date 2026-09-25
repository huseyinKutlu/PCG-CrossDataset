#!/usr/bin/env bash
# =============================================================================
# 01_download_datasets.sh
# PCG Q1 Manuscript Çalışması — Hafta 1, Adım 1
# Hocam: Bu scripti kendi makinenizde çalıştırın.
# Tahmini disk: ~12-15 GB (her iki dataset toplam)
# Tahmini süre: bağlantınıza göre 30-90 dk
# =============================================================================

set -euo pipefail

# ----- Klasör yapısı -----
PROJECT_ROOT="$HOME/pcg_project"   # İsterseniz başka bir yola değiştirin
DATA_DIR="$PROJECT_ROOT/data"
RAW_DIR="$DATA_DIR/raw"
CINC2016_DIR="$RAW_DIR/cinc2016"
CIRCOR2022_DIR="$RAW_DIR/circor2022"

mkdir -p "$CINC2016_DIR" "$CIRCOR2022_DIR"

echo ">>> Proje kök: $PROJECT_ROOT"
echo ">>> Veri yolu: $DATA_DIR"

# ----- 1) PhysioNet/CinC Challenge 2016 (eğitim seti) -----
echo ""
echo ">>> [1/2] CinC 2016 indiriliyor (training set + validation)..."
echo "    Kaynak: https://physionet.org/content/challenge-2016/"

cd "$CINC2016_DIR"

# PhysioNet resmi wget yöntemi (mirror):
wget -r -N -c -np --cut-dirs=4 -nH \
     https://physionet.org/files/challenge-2016/1.0.0/

echo ">>> CinC 2016 indirildi. İçerik:"
ls -la

# ----- 2) PhysioNet CirCor DigiScope Challenge 2022 (dış doğrulama) -----
echo ""
echo ">>> [2/2] CirCor 2022 indiriliyor (training set, public)..."
echo "    Kaynak: https://physionet.org/content/circor-heart-sound/"

cd "$CIRCOR2022_DIR"

wget -r -N -c -np --cut-dirs=4 -nH \
     https://physionet.org/files/circor-heart-sound/1.0.3/

echo ">>> CirCor 2022 indirildi. İçerik:"
ls -la

# ----- Disk kullanımı raporu -----
echo ""
echo ">>> İndirme tamamlandı. Disk kullanımı:"
du -sh "$CINC2016_DIR" "$CIRCOR2022_DIR"

echo ""
echo ">>> SONRAKİ ADIM: 02_inventory.py scriptini çalıştırın."
echo ">>> Çıktıyı (terminal log + üretilen CSV) Claude'la paylaşın."
