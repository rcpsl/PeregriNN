#!/bin/bash
TOOL_NAME=PeregriNN
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4

TOOL_DIR=$(dirname $(dirname $(realpath $0)))
VNN_SCRIPTS_DIR=$(realpath $0)
cd $TOOL_DIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate peregrinn
echo "Running network simplification script"
python vnn_scripts/simplify_network.py $CATEGORY $ONNX_FILE


pkill -9 python
cd -