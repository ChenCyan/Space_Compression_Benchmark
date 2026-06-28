#!/bin/bash
# Setup script for the HyCASS compression benchmark.
# Run on the server (source torch_env first, then this script).
# Usage:
#   source /data/cyl/miniconda3/etc/profile.d/conda.sh
#   conda activate torch_env
#   bash setup_benchmark.sh

set -euo pipefail

echo "=== Installing project dependencies ==="
pip install -r requirements.txt

echo "=== Installing sklearn (needed for KLT+DWT codec) ==="
pip install scikit-learn

echo "=== Cloning NTNU CCSDS-123.0-B-2 verification model ==="
NTNU_DIR="/data/cyl/space_compression/ccsds123_i2_hlm"
if [ ! -d "$NTNU_DIR" ]; then
    git clone https://github.com/NTNU-SmallSat-Lab/ccsds123_issue_2_verification_model.git "$NTNU_DIR"
    pip install -r "$NTNU_DIR/requirements.txt"
else
    echo "  $NTNU_DIR already exists, skipping clone"
fi

echo "=== Done! ==="
echo ""
echo "To run the benchmark:"
echo "  python -m benchmark.runner \\"
echo "      --dataset ./datasets/berlin-urban-gradient/ \\"
echo "      --dataset-type berlin \\"
echo "      --codecs jpeg2000_lossless jpeg2000_cr4 ccsds123_lossless klt_dwt_nc28_cr1 klt_dwt_nc56_cr4 \\"
echo "      --output results/benchmark.csv"
echo ""
echo "To generate plots:"
echo "  python -m benchmark.plots results/benchmark.csv --out-dir results/"
