from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *
from multiprocessing import Process, Value

eps = 1E-10

class TimeOutException(Exception):
    def __init__(self, *args, **kwargs):
        pass

def print_summary(network,prop, safety, time):
    print(network[14:19],prop,safety,time)
def alarm_handler(signum, frame):
    raise TimeOutException()

def check_property(network,x):
    u = network.evaluate(x)
    if(np.argmax(u) != target):
        # print("Potential CE succeeded")
        return True
    return False

def check_prop_samples(nn,samples):
    outs = nn.evaluate(samples.T).T
    outs = np.argmax(outs,axis = 1)
    return np.any(outs  != target)

def run_instance(network, input_bounds, check_property, target,adv_found):

    try:
        solver = Solver(network = network,property_check=check_property,target = target)
        # input_vars = [solver.state_vars[i] for i in range(len(solver.state_vars))]
        A = np.eye(network.image_size)
        lower_bound = input_bounds[:,0]
        upper_bound = input_bounds[:,1]
        solver.add_linear_constraints(A,solver.in_vars_names,lower_bound,GRB.GREATER_EQUAL)
        solver.add_linear_constraints(A,solver.in_vars_names,upper_bound,GRB.LESS_EQUAL)
        
        output_vars = [solver.out_vars[i] for i in range(len(solver.out_vars))]
        A = np.zeros(network.output_size)
        A[out_idx] = 1
        A[target] = -1
        b = [eps]
        solver.add_linear_constraints([A],solver.out_vars_names,b,GRB.GREATER_EQUAL)
        
        solver.preprocessing = False
        vars,status = solver.solve()
        if(status == 'SolFound'):
            adv_found.value = 1
        return status
        # print('Terminated')
    except Exception as e:
        raise e


if __name__ == "__main__":

    if(len(sys.argv) < 3):
        print("Arguments missing vnn.py network epsilon")
        sys.exit()
    
    TIMEOUT= 300
    adv = non_adv = timed_out = 0

    #init Neural network
    # nnet = 'VNN/mnist-net_256x4.nnet'
    nnet = sys.argv[1]
    nn = NeuralNetworkStruct()
    nn.parse_network(nnet,type = 'mnist')
    # print('Loaded network:',nnet)

    # image_files = sorted(glob.glob('images/*'))
    
    num_test = 25
    image_files = ['VNN/mnist_images/image%d'%idx for idx in range(1,num_test+1)]
    delta = float(sys.argv[-1])
    begin_time = time()
    results = []
    for im_idx, image_file in enumerate(image_files):
        with open(image_file,'r') as f:
            nn.parse_network(nnet,type = 'mnist')
            image = f.readline().split(',')
            image = np.array([float(num) for num in image[:-1]]).reshape((-1,1))/255.0
            output = nn.evaluate(image)
            target = np.argmax(output)
            nn.set_target(target)
            other_ouputs = [i for i in range(nn.output_size) if i != target]
            # print('Testing',image_file)
            # print('Output:',output,'\nTarget-->',target)
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(TIMEOUT)
        try:
            processes = []
            start_time = time()
            # print('Norm:',delta)
            #Solve the problem for each other output
            lb = np.maximum(image-delta,0.0)
            ub = np.minimum(image+delta,1.0)
            input_bounds = np.concatenate((lb,ub),axis = 1)
            samples = sample_network(nn,input_bounds,15000)
            SAT = check_prop_samples(nn,samples)
            if(SAT):
                adv +=1
                print_summary(nnet,im_idx+1,'unsafe/samples',time()-start_time)
                signal.alarm(0)
                continue
            nn.set_bounds(input_bounds)
            out_list_ub = copy(nn.layers[nn.num_layers-1]['conc_ub'])
            other_ouputs = np.flip(np.argsort(out_list_ub,axis = 0)).flatten().tolist()
            other_ouputs = [idx for idx in other_ouputs if idx!= target and out_list_ub[idx] > 0]
            adv_found = Value('i',0)
            result = ''
            for out_idx in other_ouputs:
                if 0 > nn.layers[len(nn.layers)-1]['conc_ub'][out_idx]:
                    continue
                network = deepcopy(nn)
                result = run_instance(network, input_bounds, check_property, target,adv_found)
                if(result == 'SolFound'):
                    break
                # p = Process(target=run_instance, args=(network, input_bounds, check_property, target,adv_found))
                # p.start()
                # processes.append(p)
            signal.alarm(0)
            if(result == 'SolFound'):
                #print("Adv found")
                adv +=1
                results.append("UNSAFE")
                print_summary(nnet,im_idx+1,'unsafe',time() - start_time)
            else:
                results.append("Safe")
                non_adv +=1
                print_summary(nnet,im_idx+1,'safe',time()-start_time)
            continue

            prev_n_alive = -1
            while(any(p.is_alive() for p in processes) and adv_found.value == 0):
                sleep(1)
                n_alive = np.sum([p.is_alive() for p in processes])
                if(n_alive != prev_n_alive):
                    prev_n_alive = n_alive

            if(adv_found.value == 1):
                #print("Adv found")
                adv +=1
                results.append("UNSAFE")
                print_summary(nnet,im_idx+1,'unsafe',time() - start_time)
                for p in processes:
                    p.terminate()
            else:
                #print("No Adv")
                results.append("Safe")
                non_adv +=1
                print_summary(nnet,im_idx+1,'safe',time()-start_time)

        except Exception as e:
            timed_out += 1
            print_summary(nnet,im_idx+1,'timeout',TIMEOUT)
            results.append("Timeout") 
            continue
        for p in processes:
            p.terminate()
    print('eps:',delta,',Adv:',adv,',non_adv:',non_adv,',unproven:',timed_out,',Total time:',time() - begin_time)



        

#active neurons [ 3  4  6  7  9 12 15 20 22 23 25 27 28 30 32 33 34 40 45 49]

