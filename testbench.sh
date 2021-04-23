#!/bin/bash


#Timeout in seconds
TIMEOUT=300

# make results folder and clear results
mkdir -p ./results
rm ./results/*

for net in 2 4 6
do 
    net_dir="VNN/mnist-net_256x${net}.nnet"
	ep=0.02
    printf "\n===========\nChecking network mnist-net_256x${net}.nnet with epsilon ${ep} and timeout ${TIMEOUT}\n"
    for im_idx in {1..25}
    do
        image="VNN/mnist_images/image${im_idx}"
        #use >> instead of | tee -a if you don't want the results to print on the terminal too
        python3 -u peregriNN.py $net_dir $image $ep --timeout $TIMEOUT | tee -a results/result_256X${net}_${ep}.txt || { echo "Exiting script"; exit 1; }
    done
done

net=2
ep=0.05
net_dir="VNN/mnist-net_256x${net}.nnet"
printf "\n===========\nChecking network mnist-net_256x${net}.nnet with epsilon ${ep} and timeout ${TIMEOUT}\n"
    for im_idx in {1..25}
    do
        image="VNN/mnist_images/image${im_idx}"
        #use >> instead of | tee -a if you don't want the results to print on the terminal too
        python3 -u peregriNN.py $net_dir $image $ep --timeout $TIMEOUT | tee -a results/result_256X${net}_${ep}.txt || { echo "Exiting script"; exit 1; }
    done
