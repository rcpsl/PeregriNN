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
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running $TOOL_NAME on benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

TOOL_DIR=$(dirname $(dirname $(realpath $0)))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate peregrinn

# run the tool to produce the results file
python $TOOL_DIR/peregriNN.py "$ONNX_FILE" "$VNNLIB_FILE" --timeout "$TIMEOUT" --result_file "$RESULTS_FILE" --category $CATEGORY