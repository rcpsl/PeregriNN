import numpy as np
import constant
import pickle
import sys
from Workspace import *
from keras.models import load_model
from copy import copy
np.random.seed(50)
class NeuralNetworkStruct(object):

    def __init__(self ,layers_sizes=[], load_weights = False, input_bounds = None):
        # num_lasers includes the output layer
        if(len(layers_sizes) == 0):
            return
        self.num_layers = len(layers_sizes)
        self.image_size  = layers_sizes[0]
        self.output_size = layers_sizes[-1]
        self.num_hidden_neurons = sum(layers_sizes[1:-1])
        self.layers_sizes = layers_sizes
        self.input_min = np.zeros(self.image_size)
        self.input_max = np.zeros(self.image_size)
        self.input_mean = np.zeros(self.image_size)
        self.input_range = np.zeros(self.image_size)
        self.out_mean = 0
        self.out_range = 0
        input_bound = input_bounds
        if(input_bounds is None):
            input_bound = np.ones((self.layers_sizes[0]+1,2))
            input_bound[:-1,0] = -1E100
            input_bound[:-1,1] = 1E100
        
        self.layers = {}
        if(load_weights):
            self.model = load_model("model/my_model.h5")

        #input layer
        in_bound = input_bound[:-1,:]
        self.layers[0] = {'num_nodes':self.image_size, 'weights': [], 'type':'input','lb':in_bound[:,0].reshape((-1,1)),
                        'ub':in_bound[:,1].reshape((-1,1)),
                        'Relu_lb': in_bound[:,0].reshape((-1,1)), 'Relu_ub': in_bound[:,1].reshape((-1,1))}

        for index in range(self.num_layers):
            if(index == 0):
                continue
            self.layers[index]  = {'num_nodes': layers_sizes[index], 'weights': []}
            self.layers[index]['type'] = 'hidden'

            if load_weights:
                self.layers[index]['weights'] = self.model.get_weights()[2*index].T
                self.layers[index]['bias'] = self.model.get_weights()[2*index + 1]


            else:
                self.layers[index]['weights'] = np.random.normal(scale=2.0, size=(layers_sizes[index], layers_sizes[index-1]))
                self.layers[index]['bias'] = np.random.normal(scale=0.5, size=(layers_sizes[index],1))

        # self.__compute_IA_bounds()
        # self.__compute_sym_bounds()
        self.layers[self.num_layers-1]['type'] = 'output'

    def __compute_sym_bounds(self):
        #first layer Symbolic interval
        W = self.layers[1]['weights']
        b = self.layers[1]['bias'].reshape((-1,1))
        input_bounds = np.hstack((self.layers[0]['lb'],self.layers[0]['ub']))
        input_bounds = np.vstack((input_bounds,np.ones(2)))
        input_sym = SymbolicInterval(np.hstack((W,b)),np.hstack((W,b)),input_bounds)
        self.layers[1]['sym_lb'] = input_sym.concrete_Mlower_bound(input_sym.lower,input_sym.interval)
        self.layers[1]['sym_ub'] = input_sym.concrete_Mupper_bound(input_sym.upper,input_sym.interval)
        input_sym = input_sym.forward_relu(input_sym)
        for layer_idx,layer in self.layers.items():
            if(layer_idx < 2):
                continue
            weights = (layer['weights'],layer['bias'])
            input_sym = input_sym.forward_linear(weights)
            layer['sym_lb'] = input_sym.concrete_Mlower_bound(input_sym.lower,input_sym.interval)
            layer['sym_ub'] = input_sym.concrete_Mupper_bound(input_sym.upper,input_sym.interval)
            if(layer['type'] == 'hidden'):
                input_sym = input_sym.forward_relu(input_sym)


    def __compute_IA_bounds(self):
        for index in range(self.num_layers):
            
            if(self.layers[index]['type'] != 'input'):
                W = self.layers[index]['weights']
                b = self.layers[index]['bias']
                prev_lb = self.layers[index-1]['Relu_lb']
                prev_ub = self.layers[index-1]['Relu_ub']
                self.layers[index]['lb'] = (np.maximum(0,W).dot(prev_lb) + np.minimum(0,W).dot(prev_ub) + b).reshape((-1,1))
                self.layers[index]['ub'] = (np.maximum(0,W).dot(prev_ub) + np.minimum(0,W).dot(prev_lb) + b).reshape((-1,1))
                if(self.layers[index]['type'] is not 'output'):
                    self.layers[index]['Relu_lb'] = np.maximum(0,self.layers[index]['lb']).reshape((-1,1))
                    self.layers[index]['Relu_ub'] = np.maximum(0,self.layers[index]['ub']).reshape((-1,1))

    def set_weights(self,Weights,biases):

        for index in range(self.num_layers):
            if(index == 0):
                continue
            self.layers[index]['weights'] = Weights[index - 1]
            self.layers[index]['bias'] = biases[index - 1].reshape((-1,1))
        
        self.__compute_IA_bounds()
        self.__compute_sym_bounds()


    def __set_stats(self,stats):

        self.input_min = np.array(stats['min'])
        self.input_max = np.array(stats['max'])
        self.input_mean = np.array(stats['mean'][:-1])
        self.input_range = np.array(stats['range'][:-1])
        self.out_mean = stats['mean'][-1]
        self.out_range = stats['range'][-1]
    
    def set_bounds(self,input_bounds):
        self.layers[0]['lb'] = input_bounds[:,0].reshape((-1,1))
        self.layers[0]['ub'] = input_bounds[:,1].reshape((-1,1))
        self.layers[0]['Relu_lb'] = input_bounds[:,0].reshape((-1,1))
        self.layers[0]['Relu_ub'] = input_bounds[:,1].reshape((-1,1))
        self.__compute_IA_bounds()
        self.__compute_sym_bounds()


    def evaluate(self,input):
        prev = input
        for index in range(self.num_layers):
            if(index == 0):
                continue
            W = self.layers[index]['weights']
            b = self.layers[index]['bias']
            net = W.dot(prev) + b
            if(self.layers[index]['type'] == 'output'):
                prev =  net
            else:
                prev = np.maximum(0,net)
        return prev
    def normalize_input(self,inputIndex,val):
        in_min = self.input_min[inputIndex]
        in_max = self.input_max[inputIndex]
        in_mean = self.input_mean[inputIndex]
        in_range = self.input_range[inputIndex]
        if ( val < in_min ):
            val = in_min
        elif ( val > in_max ):
            val = in_max
        return ( val - in_mean ) / in_range
    def unnormalize_input(self,inputIndex, val):
        in_mean = self.input_mean[inputIndex]
        in_range = self.input_range[inputIndex]
        return  (val * in_range) + in_mean
        
    def parse_network(self, model_file, weights_file):
        with open(model_file,'rb') as f:
            model_fmt_file = f.readlines() 
            layers_sizes = list(map(int,model_fmt_file[4][:-2].split(','))) 
            f.close()
        
        with open(weights_file,'rb') as f:
            weights_arr = np.load(weights_file)
            f.close()
        
        num_weights = 0
        for idx in range(len(layers_sizes) - 1):
            num_weights += (layers_sizes[idx] * layers_sizes[idx+1])
        weights_strt_idx = 0
        bias_strt_idx = num_weights
        W = []
        biases = []
        for idx in range(len(layers_sizes) - 1):
            source = layers_sizes[idx]
            target = layers_sizes[idx + 1]
            W_layer = weights_arr[weights_strt_idx: weights_strt_idx + (source*target)]
            b = np.array(weights_arr[bias_strt_idx:bias_strt_idx + target])
            W.append(np.array(W_layer).reshape((target,source)))
            biases.append(b)
            weights_strt_idx += source*target
            bias_strt_idx   += target

        #Read min and max for inputs
        mins = list(map(float,model_fmt_file[6][:-2].split(','))) 
        maxs = list(map(float,model_fmt_file[7][:-2].split(','))) 
        means = list(map(float,model_fmt_file[8][:-2].split(','))) 
        ranges = list(map(float,model_fmt_file[9][:-2].split(','))) 
        stats = {'min' :mins, 'max':maxs,'mean':means,'range':ranges}
        self.__init__(layers_sizes)
        self.set_weights(W,biases)
        self.__set_stats(stats)  
        # return layers_sizes,W,biases,stats

        

class SymbolicInterval(object):
    
    def __init__(self, low, upp, interval = None):

        self.lower = low
        self.upper = upp
        
        if(interval is not None):
            self.interval = interval
        else:
            self.interval = np.zeros((self.lower.shape[1]-1,2))
        
    def forward_linear(self, weights):
        W,b = weights
        out_upp = np.atleast_2d(np.matmul(np.maximum(W,0),self.upper) + np.matmul(np.minimum(W,0),self.lower))
        out_low = np.atleast_2d(np.matmul(np.maximum(W,0),self.lower) + np.matmul(np.minimum(W,0),self.upper))
        out_upp[:,-1] += b.flatten()
        out_low[:,-1]+= b.flatten()
        return SymbolicInterval(out_low,out_upp,self.interval)

    def forward_relu(self, symInterval):
        relu_lower_equtions = copy(symInterval.lower)
        relu_upper_equations = copy(symInterval.upper)
        for row in range(relu_lower_equtions.shape[0]):
            relu_lower_eq = relu_lower_equtions[row]
            relu_upper_eq = relu_upper_equations[row]
            lower_lb = self.concrete_lower_bound(relu_lower_eq, symInterval.interval)
            upper_lb = self.concrete_lower_bound(relu_upper_eq, symInterval.interval)
            upper_ub = self.concrete_upper_bound(relu_upper_eq, symInterval.interval)


            if(lower_lb >= 0):
                pass
            elif(upper_ub <= 0):
                relu_lower_eq[:]    = 0
                relu_upper_eq[:]    = 0
            else:
                relu_lower_eq[:]    = 0
                if(upper_lb < 0):
                    relu_upper_eq[:]   = np.hstack((np.zeros(len(relu_upper_eq) - 1), upper_ub))
        
        return SymbolicInterval(relu_lower_equtions,relu_upper_equations, self.interval)


    def concrete_lower_bound(self, equation, interval):
        #Get indices of coeff >0
        p_idx = np.where(equation > 0)[0]
        n_idx = np.where(equation <= 0)[0]
        lb = equation[p_idx].dot(interval[p_idx,0]) + equation[n_idx].dot(interval[n_idx,1])        

        return lb

    def concrete_upper_bound(self, equation, interval):
        p_idx = np.where(equation > 0)[0]
        n_idx = np.where(equation <= 0)[0]       
        ub = equation[p_idx].dot(interval[p_idx,1]) + equation[n_idx].dot(interval[n_idx,0])
        return ub

    def concrete_Mlower_bound(self, equations, interval):
        lb = []
        for equation in equations:
            lb.append(self.concrete_lower_bound(equation,interval))

        return np.array(lb).reshape((-1,1))

    def concrete_Mupper_bound(self, equations, interval):
        ub = []
        for equation in equations:
            ub.append(self.concrete_upper_bound(equation,interval))
        return np.array(ub).reshape((-1,1))





if __name__ == "__main__":
    layers_sizes = [2,2,1]
    input_bounds = np.array([[1,10],[1,2],[1,1]])

    nn= NeuralNetworkStruct(layers_sizes,input_bounds=input_bounds)
    weights = []
    biases = []
    weights.append(np.array([[1,2],[2,1]]))
    biases.append(np.array([0,0]))
    weights.append(np.array([1,-1]))
    biases.append(np.array([0]))
    nn.set_weights(weights,biases)
    pass

