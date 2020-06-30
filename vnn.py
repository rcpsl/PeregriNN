from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *
from multiprocessing import Process, Value

eps = 5E-1

class TimeOutException(Exception):
    pass

 
def alarm_handler(signum, frame):
    print('TIMEOUT!')
    raise TimeOutException()

def check_property(x):
    u = nn.evaluate(x)
    if(np.argmax(u) != target):
        print("Potential CE succeeded")
        return True
    return False


def run_instance(network, input_bounds, check_property, target,adv_found):

    try:
        solver = Solver(network = network,property_check=check_property,target = target)
        input_vars = [solver.state_vars[i] for i in range(len(solver.state_vars))]
        A = np.eye(network.image_size)
        lower_bound = input_bounds[:,0]
        upper_bound = input_bounds[:,1]
        solver.add_linear_constraints(A,input_vars,lower_bound,GRB.GREATER_EQUAL)
        solver.add_linear_constraints(A,input_vars,upper_bound,GRB.LESS_EQUAL)
        
        output_vars = [solver.out_vars[i] for i in range(len(solver.out_vars))]
        A = np.zeros(network.output_size)
        A[out_idx] = 1
        A[target] = -1
        b = [eps]
        solver.add_linear_constraints([A],output_vars,b,GRB.GREATER_EQUAL)
        
        solver.preprocessing = False
        nn_in,nn_out,status = solver.solve()
        if(status == 'SolFound'):
            adv_found.value = 1
        
        # print('Terminated')
    except Exception as e:
        print(e)


if __name__ == "__main__":

    TIMEOUT= 1200
    adv = non_adv = timed_out = 0

    #init Neural network
    nnet = 'VNN/mnist-net_256x6.nnet'
    nn = NeuralNetworkStruct()
    nn.parse_network(nnet,type = 'mnist')
    print('Loaded network:',nnet)

    # image_files = sorted(glob.glob('images/*'))
    
    num_test = 50
    image_files = ['VNN/mnist_images/image%d'%idx for idx in range(1,num_test+1)]
    deltas = [0.03]
    begin_time = time()
    for image_file in image_files:
        start_time = time()
        with open(image_file,'r') as f:
            nn.parse_network(nnet,type = 'mnist')
            image = f.readline().split(',')
            image = np.array([float(num) for num in image[:-1]]).reshape((-1,1))/255.0
            output = nn.evaluate(image)
            target = np.argmax(output)
            nn.set_target(target)
            other_ouputs = [i for i in range(nn.output_size) if i != target]
            print('Testing',image_file)
            print('Output:',output,'\nTarget-->',target)
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(TIMEOUT)
        try:
            for delta in deltas:
                print('Norm:',delta)
                #Solve the problem for each other output
                adv_found = False
                lb = np.maximum(image-delta,0.0)
                ub = np.minimum(image+delta,1.0)
                input_bounds = np.concatenate((lb,ub),axis = 1)
                nn.set_bounds(input_bounds)
                out_list_ub = copy(nn.layers[nn.num_layers-1]['conc_ub'])
                other_ouputs = np.flip(np.argsort(out_list_ub,axis = 0))
                other_ouputs = [idx for idx in other_ouputs if idx!= target and out_list_ub[idx] > 0]
                adv_found = Value('i',0)
                processes = []
                for out_idx in other_ouputs:
                    network = deepcopy(nn)
                    p = Process(target=run_instance, args=(network, input_bounds, check_property, target,adv_found))
                    p.start()
                    processes.append(p)
                
                while(any(p.is_alive() for p in processes) and adv_found.value == 0):
                    sleep(1)
                if(adv_found.value == 1):
                    print("Adv found")
                    adv +=1
                else:
                    print("No Adv")
                    non_adv +=1
                for p in processes:
                    p.terminate()
        except TimeOutException as e:
            timed_out += 1
    print('Adv:',adv,',non_adv:',non_adv,',unproven:',timed_out,',Total time:',time() - begin_time)



        

#active neurons [ 3  4  6  7  9 12 15 20 22 23 25 27 28 30 32 33 34 40 45 49]

