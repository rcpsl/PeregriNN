This folder contains three fully-connected ReLU networks trained with
the MNIST dataset. 

Models are provided in both nnet and ONNX format.  Input images'
pixels are assumed to be floats between 0 and 1. No further
normalisation is required, so lines 4-8 in the nnet files can be
disregarded. 

Suggested l_infty constrained permutations for local robustness
benchmarks are:

0.01: Easy (Mostly safe)

0.03: Difficult (Mixed safe and unsafe)

0.05: Intermediate to Difficult (Mostly unsafe) 