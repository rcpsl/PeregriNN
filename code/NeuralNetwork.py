import numpy as np
import constant
import pickle
import sys
from Workspace import *
from copy import copy
np.random.seed(50)
eps = 1E-5
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
        self.input_bound = input_bounds
        self.nonlin_relus = []
        self.active_relus = []
        self.inactive_relus = []
        if(input_bounds is None):
            self.input_bound = np.ones((self.layers_sizes[0]+1,2))
            self.input_bound[:-1,0] = -1E10
            self.input_bound[:-1,1] = 1E10
        
        self.layers = {}
        if(load_weights):
            self.model = load_model("model/my_model.h5")

        #input layer
        in_bound = self.input_bound[:-1,:]
        self.layers[0] = {'idx':0, 'num_nodes':self.image_size, 'weights': [], 'type':'input','lb':in_bound[:,0].reshape((-1,1)),
                        'ub':in_bound[:,1].reshape((-1,1)),
                        'Relu_lb': in_bound[:,0].reshape((-1,1)), 'Relu_ub': in_bound[:,1].reshape((-1,1))}

        for index in range(self.num_layers):
            if(index == 0):
                continue
            self.layers[index]  = {'idx':index, 'num_nodes': layers_sizes[index], 'weights': []}
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
    def recompute_bounds(self,layers_mask):
        self.nonlin_relus = []
        self.active_relus = []
        self.inactive_relus = []
        I = np.zeros((self.image_size ,self.image_size+ 1))
        np.fill_diagonal(I,1)
        layer_sym = SymbolicInterval(I,I,self.input_bound)
        for layer_idx in range(1,len(self.layers)):
            layer = self.layers[layer_idx]

            weights = (layer['weights'],layer['bias'])
            layer_sym = layer_sym.forward_linear(weights)
            layer['in_sym'] = layer_sym
            layer['in_lb'] = layer_sym.concrete_Mlower_bound(layer_sym.lower,layer_sym.interval)
            layer['in_ub'] = layer_sym.concrete_Mupper_bound(layer_sym.upper,layer_sym.interval)
            if(layer['type'] == 'hidden'):
                layer_mask = layers_mask[layer_idx-1]
                active_neurons = np.where(layer_mask == 1)[0]
                self.active_relus += [[layer_idx,idx] for idx in active_neurons]
                inactive_neurons = np.where(layer_mask == 0)[0]
                self.inactive_relus += [[layer_idx,idx] for idx in inactive_neurons]
                layer_sym,error_vec = layer_sym.forward_relu(layer = layer_idx,nonlin_relus = self.nonlin_relus,inact_relus=self.inactive_relus,act_relus= self.active_relus)
                layer_sym.lower[active_neurons]  = layer_sym.upper[active_neurons] =  layer['in_sym'].upper[active_neurons]
                layer_sym.upper[inactive_neurons] = layer_sym.lower[inactive_neurons] = 0      
            layer['Relu_sym'] = layer_sym
            layer['conc_lb'] = layer_sym.concrete_Mlower_bound(layer_sym.lower,layer_sym.interval)
            layer['conc_ub'] = layer_sym.concrete_Mupper_bound(layer_sym.upper,layer_sym.interval)
        sorted(self.nonlin_relus)

    def __compute_sym_bounds(self):
        #first layer Symbolic interval
        self.nonlin_relus = []
        W = self.layers[1]['weights']
        b = self.layers[1]['bias'].reshape((-1,1))
        input_bounds = np.hstack((self.layers[0]['lb'],self.layers[0]['ub']))
        input_bounds = np.vstack((input_bounds,np.ones(2)))
        input_sym = SymbolicInterval(np.hstack((W,b)),np.hstack((W,b)),input_bounds)
        self.layers[1]['in_sym'] = input_sym
        self.layers[1]['in_lb'] = input_sym.concrete_Mlower_bound(input_sym.lower,input_sym.interval)
        self.layers[1]['in_ub'] = input_sym.concrete_Mupper_bound(input_sym.upper,input_sym.interval)
        # self.layers[1]['Relu_sym'] = input_sym
        input_sym,error_vec = input_sym.forward_relu(layer = 1,nonlin_relus = self.nonlin_relus,inact_relus=self.inactive_relus,act_relus= self.active_relus)
        self.layers[1]['conc_lb'] = input_sym.concrete_Mlower_bound(input_sym.lower,input_sym.interval)
        self.layers[1]['conc_ub'] = input_sym.concrete_Mupper_bound(input_sym.upper,input_sym.interval)
        self.layers[1]['Relu_sym'] = input_sym
        for layer_idx,layer in self.layers.items():
            if(layer_idx < 2):
                continue
            weights = (layer['weights'],layer['bias'])
            input_sym = input_sym.forward_linear(weights)
            layer['in_lb'] = input_sym.concrete_Mlower_bound(input_sym.lower,input_sym.interval)
            layer['in_ub'] = input_sym.concrete_Mupper_bound(input_sym.upper,input_sym.interval)
            layer['in_sym'] = input_sym
            if(layer['type'] == 'hidden'):
                input_sym,error_vec = input_sym.forward_relu(layer = layer_idx,nonlin_relus = self.nonlin_relus, inact_relus=self.inactive_relus,act_relus= self.active_relus)
            layer['Relu_sym'] = input_sym
            layer['conc_lb'] = input_sym.concrete_Mlower_bound(input_sym.lower,input_sym.interval)
            layer['conc_ub'] = input_sym.concrete_Mupper_bound(input_sym.upper,input_sym.interval)

        sorted(self.nonlin_relus) 
    def update_bounds(self,layer_idx,neuron_idx,bounds,layers_mask = None):
        input_sym = self.layers[layer_idx]['Relu_sym']
        if(np.all(bounds[0] - input_sym.lower <= eps) and np.all(bounds[1] - input_sym.upper <= eps)):
            return
        input_sym.lower[neuron_idx] = bounds[0]
        input_sym.upper[neuron_idx] = bounds[1]
        self.layers[layer_idx]['conc_lb'][neuron_idx] = input_sym.concrete_lower_bound(input_sym.lower[neuron_idx],input_sym.interval)
        self.layers[layer_idx]['conc_ub'][neuron_idx] = input_sym.concrete_upper_bound(input_sym.upper[neuron_idx],input_sym.interval)

        for idx,layer in self.layers.items():
            if(idx < layer_idx + 1):
                continue
            if(layers_mask is None):
                mask = 1
            else:
                mask = layers_mask[idx-1]
            weights = (layer['weights'],layer['bias'])
            input_sym = input_sym.forward_linear(weights)
            layer['in_lb'] = input_sym.concrete_Mlower_bound(input_sym.lower,input_sym.interval)
            layer['in_ub'] = input_sym.concrete_Mupper_bound(input_sym.upper,input_sym.interval)
            if(layer['type'] == 'hidden'):
                input_sym,error_vec = input_sym.forward_relu(input_sym)
                input_sym.lower *= mask
                input_sym.upper *= mask
            layer['Relu_sym'] = input_sym
            layer['conc_lb'] = input_sym.concrete_Mlower_bound(input_sym.lower,input_sym.interval) 
            layer['conc_ub'] = input_sym.concrete_Mupper_bound(input_sym.upper,input_sym.interval)



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
        self.input_bound = input_bounds
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
    def normalize_input(self,val):
        ret = np.zeros_like(val)
        for inputIndex in range(len(val)):
            in_min = self.input_min[inputIndex]
            in_max = self.input_max[inputIndex]
            in_mean = self.input_mean[inputIndex]
            in_range = self.input_range[inputIndex]
            if ( val[inputIndex] < in_min ):
                ret[inputIndex] = in_min
            elif ( val[inputIndex] > in_max ):
                ret[inputIndex] = in_max
            else:
                ret[inputIndex] = ( val[inputIndex] - in_mean ) / in_range
        return ret
    def unnormalize_input(self,inputIndex, val):
        in_mean = self.input_mean[inputIndex]
        in_range = self.input_range[inputIndex]
        return  (val * in_range) + in_mean
        
    def parse_network(self, model_file,type = 'Acas'):
        with open(model_file,'r') as f:
            start_idx = 4
            if(type == 'mnist'):
                start_idx = 2
            model_fmt_file = f.readlines() 
            layers_sizes = list(map(int,model_fmt_file[start_idx][:-2].split(','))) 
            f.close()
        
        W = []
        biases =[]
        start_idx = 10
        if(type == 'mnist'):
            start_idx = 3
        for idx in range(1, len(layers_sizes)):
            source = layers_sizes[idx-1]
            target = layers_sizes[idx]
            layer_weights = np.zeros((target,source))
            layer_bias = np.zeros(target)
            for row in range(target):
               weights = np.array(list(map(float,model_fmt_file[start_idx].split(',')[:-1])))
               layer_weights[row] = weights
               start_idx +=1
            for row in range(target):
                bias = float(model_fmt_file[start_idx].split(',')[0])
                layer_bias[row] = bias
                start_idx +=1
            W.append(layer_weights)
            biases.append(layer_bias)
        
        #Read min and max for inputs
        mins = list(map(float,model_fmt_file[6].split(',')[:-1])) 
        maxs = list(map(float,model_fmt_file[7].split(',')[:-1])) 
        means = list(map(float,model_fmt_file[8].split(',')[:-1])) 
        ranges = list(map(float,model_fmt_file[9].split(',')[:-1])) 
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

    def forward_relu(self,layer = -1,nonlin_relus = [],inact_relus = [],act_relus = []):
        relu_lower_equtions = copy(self.lower)
        relu_upper_equations = copy(self.upper)
        error_vec = np.zeros(len(relu_lower_equtions))
        for row in range(relu_lower_equtions.shape[0]):
            relu_lower_eq = relu_lower_equtions[row]
            relu_upper_eq = relu_upper_equations[row]
            lower_lb = self.concrete_lower_bound(relu_lower_eq, self.interval)
            lower_ub = self.concrete_upper_bound(relu_lower_eq, self.interval)
            upper_lb = self.concrete_lower_bound(relu_upper_eq, self.interval)
            upper_ub = self.concrete_upper_bound(relu_upper_eq, self.interval)


            if(lower_lb >= 0):
                act_relus.append([layer,row])
            elif(upper_ub <= 0):
                relu_lower_eq[:]    = 0
                relu_upper_eq[:]    = 0
                inact_relus.append([layer,row])
            else:
                nonlin_relus.append([layer,row])
                if(lower_ub >eps):
                    relu_lower_eq[:]    =  lower_ub * (relu_lower_eq) / (lower_ub - lower_lb)
                else:
                    relu_lower_eq[:] = 0

                if(upper_lb < eps):
                    relu_upper_eq[:]   = upper_ub * (relu_upper_eq) / (upper_ub - upper_lb)
                    relu_upper_eq[-1]  -= upper_ub* upper_lb / (upper_ub - upper_lb)
                    error_vec[row] -= upper_ub* upper_lb / (upper_ub - upper_lb)
        
        return SymbolicInterval(relu_lower_equtions,relu_upper_equations, self.interval),np.diagflat(error_vec)


    def concrete_lower_bound(self, equation, interval):
        #Get indices of coeff >0
        p_idx = np.where(equation[:-1] > 0)[0]
        n_idx = np.where(equation[:-1] <= 0)[0]
        lb = equation[p_idx].dot(interval[p_idx,0]) + equation[n_idx].dot(interval[n_idx,1]) + equation[-1]     

        return lb

    def concrete_upper_bound(self, equation, interval):
        p_idx = np.where(equation[:-1] > 0)[0]
        n_idx = np.where(equation[:-1] <= 0)[0]       
        ub = equation[p_idx].dot(interval[p_idx,1]) + equation[n_idx].dot(interval[n_idx,0]) + equation[-1]
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
    layers_sizes = [1,2,1,1]
    input_bounds = np.array([[0,1],[1,1]])

    nn= NeuralNetworkStruct(layers_sizes,input_bounds=input_bounds)
    weights = []
    biases = []
    weights.append(np.array([[2],[-1]]))
    biases.append(np.array([-1,1]))
    weights.append(np.array([-3,1]))
    biases.append(np.array([0]))
    weights.append(np.array([1]))
    biases.append(np.array([0]))
    nn.set_weights(weights,biases)
    nn.set_bounds(np.array([0,1]).reshape((1,-1)))
    pass

