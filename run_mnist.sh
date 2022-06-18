#!/bin/bash


#Timeout in seconds
TIMEOUT=300

# make results folder and clear results
mkdir -p ./results
rm ./results/*

for net in 6
do 
    net_dir="VNN/mnist-net_256x${net}.onnx"
	# 2 epsilon values
	for ep in  0.05
    do 
        printf "\n===========\nChecking network mnist-net_256x${net}.nnet with epsilon ${ep} and timeout ${TIMEOUT}\n"
        for im_idx in {15..50}
        do
            image="VNN/mnist_images/image${im_idx}"
            #use >> instead of | tee if you don't want the results to print on the terminal too
            python -u peregriNN.py $net_dir $image --eps $ep --timeout $TIMEOUT | tee -a results/result_256X${net}_${ep}.txt || { echo "Exiting script"; exit 1; }
        done
	done
done
