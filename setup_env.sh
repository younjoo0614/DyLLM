#!/bin/bash
set -e

echo "Setting up DyLLM environment..."

echo "Installing CUDA Toolkit 13.0..."
conda install cuda-toolkit cuda-version=13.0 -y

# Set CUDA paths for compilation
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "Installing PyTorch..."
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130

echo "Setting build options..."
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
export MAX_JOBS=24

echo "Installing Flash Attention..."
pip install flash-attn --no-build-isolation

echo "Building and Installing DyLLM..."
pip install -e . --no-build-isolation

echo "Setup complete! Environment '$ENV_NAME' is ready."
