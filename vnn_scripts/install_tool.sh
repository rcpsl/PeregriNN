#!/bin/bash
TOOL_NAME="PeregriNN"
VERSION_STRING="v1"

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing miniconda"
#Install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
chmod a+x ~/miniconda3/miniconda.sh
~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash zsh
source ~/.bashrc

#Install environment
echo "Installing $TOOL_NAME"
TOOL_DIR=$(dirname $(dirname $(realpath $0)))

echo "Installing Conda environment"
echo Y | conda env create -n peregrinn -f "$TOOL_DIR/environment.yml"

echo "Installing Gurobi License"
printf "y\n\n" | grbgetkey 86561fd4-018a-11ed-9256-0242ac120005

exec bash

