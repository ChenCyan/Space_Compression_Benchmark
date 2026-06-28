#!/bin/bash
# ============================================================================
# Berlin-Urban-Gradient 数据集 & HyCASS 预训练权重 准备脚本
# ============================================================================
# 在服务器 hycass 项目根目录运行:
#   cd /data/cyl/space_compression/hycass
#   bash prepare_data.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="/data/cyl/space_compression/hycass"
DATASET_DIR="$PROJECT_ROOT/datasets/berlin-urban-gradient"

echo "=========================================="
echo "  Step 1: 下载 Berlin-Urban-Gradient 数据集"
echo "=========================================="

ZIP_URL="https://box.hu-berlin.de/f/e4fa78c198bc4d868d30/?dl=1"
ZIP_PATH="$DATASET_DIR/berlin_raw.zip"

if [ -f "$ZIP_PATH" ]; then
    SIZE=$(stat -c%s "$ZIP_PATH" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 200000000 ]; then
        echo "  [SKIP] $ZIP_PATH already exists (${SIZE} bytes), looks complete."
    else
        echo "  Resuming/redownloading (current: ${SIZE} bytes, target: ~262 MB)..."
        rm -f "$ZIP_PATH"
    fi
fi

if [ ! -f "$ZIP_PATH" ]; then
    echo "  Downloading from $ZIP_URL ..."
    wget -c -O "$ZIP_PATH" "$ZIP_URL" || curl -C - -L -o "$ZIP_PATH" "$ZIP_URL"
    echo "  Download complete: $(stat -c%s "$ZIP_PATH") bytes"
fi

echo ""
echo "=========================================="
echo "  Step 2: 解压 + 预处理 (patchify + splits)"
echo "=========================================="

# Unzip
TIFF_PATH="$DATASET_DIR/raster/hymap_berlin.tif"
if [ ! -f "$TIFF_PATH" ]; then
    echo "  Unzipping ..."
    unzip -o "$ZIP_PATH" -d "$DATASET_DIR/"
    echo "  Unzip complete."
else
    echo "  [SKIP] $TIFF_PATH already exists."
fi

# Run preprocessing script (patchify + split creation)
echo "  Running patchify + split creation ..."
source /data/cyl/miniconda3/etc/profile.d/conda.sh
conda activate torch_env
python prepare_berlin_data.py "$DATASET_DIR"

echo ""
echo "=========================================="
echo "  Step 3: 下载 HyCASS 预训练权重"
echo "=========================================="
echo ""
echo "  !!! This step requires a WEB BROWSER !!!"
echo "  The Seafile page at tubcloud.tu-berlin.de uses JS and cannot"
echo "  be scraped from the command line."
echo ""
echo "  Manual steps:"
echo "    1. Open in browser: https://tubcloud.tu-berlin.de/s/2LDG47Zzcy529eG"
echo "    2. Download ALL .pth.tar files from the directory"
echo "    3. scp them to the server:"
echo "       scp /path/to/downloaded/*.pth.tar huaweiyun:$PROJECT_ROOT/pretrained/"
echo ""
echo "  If the Seafile page has a 'Download' button for the whole"
echo "  directory, download the zip and extract:"
echo "    mkdir -p $PROJECT_ROOT/pretrained"
echo "    unzip ~/Downloads/berlin-urban-gradient.zip -d $PROJECT_ROOT/pretrained/"
echo ""
echo "=========================================="
echo "  DONE"
echo "=========================================="
echo ""
echo "Next: Run the benchmark"
echo "  python -m benchmark.runner -d $DATASET_DIR/ --dataset-type berlin --codecs ..."
