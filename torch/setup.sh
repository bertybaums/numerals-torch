#!/usr/bin/env bash
# Run once on fortyfive to create the virtual environment and install PyTorch.
# Usage: bash setup.sh

source /etc/profile
set -e

module load python/3.11.11
module load cuda/12.8

VENV_DIR="$HOME/venvs/numerals"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

echo "Installing PyTorch 2.6 (CUDA 12.4 wheel) ..."
pip install torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

echo "Done. Virtual environment ready at $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
