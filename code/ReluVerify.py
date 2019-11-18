from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
from keras.models import load_model
from StateSpacePartitioner import StateSpacePartitioner
import pandas as pd
import sys,os
from multiprocessing import Queue,Lock,Pool,current_process,Manager
import glob
class KeyboardInterruptError(Exception): pass

def load_regions(dir = "./regions/"):
    filename = dir + 'abst_reg_H_rep.txt'
    with open(filename, 'rb') as inputFile:
        abst_reg_H_rep = pickle.load(inputFile)

    
    filename = dir + 'lidar_config_dict.txt'
    with open(filename, 'rb') as inputFile:
        lidar_config_dict = pickle.load(inputFile)
    return abst_reg_H_rep, lidar_config_dict

def add_initial_state_constraints(solver, regions, start_region):

    state_vars = [solver.state_vars[0],solver.state_vars[1]]
    # region  = abst_reg_H_rep[start_region]
    # region_H = partitioner.symbolic_states[start_region]['PolygonH']
    region_H = regions[start_region]
    A, b = region_H['PolygonH']['A'], region_H['PolygonH']['b']
    solver.add_linear_constraints(A, state_vars, b)

def add_final_state_constraints(solver, end_region):
    state_vars = [solver.next_state_vars[0],solver.next_state_vars[1]]
    # region  = abst_reg_H_rep[end_region]
    region_H = partitioner.symbolic_states[end_region]['PolygonH']
    A, b = region_H['A'], region_H['b']
    solver.add_linear_constraints(A, state_vars, b)

    state_vars = [solver.next_state_vars[2],solver.next_state_vars[3]]
    A = [[1.0, 0.0],[0.0,1.0],[-1.0,0.0],[0.0,-1.0]]
    solver.add_linear_constraints(A, state_vars, b)


def add_lidar_constraints(solver, start_region):
    """
    For a certain laser i, if it intersects a vertical obstacle:
        x_i = x_obstacle
        y_i = y_car + (x_obstacle - x_car) tan(laser_angle)
    Otherwise:
        x_i = x_car + (y_obstacle - y_car) cot(laser_angle)
        y_i = y_obstacle
    """

    lidar_config = lidar_config_dict[start_region]
    #print lidar_config

    for i in xrange(num_lasers):
        # NOTE: Difference between indices of x,y coordinates for the same laser in image is number of lasers
        rVars = [solver.im_vars[i], solver.im_vars[i+ num_lasers], 
                solver.state_vars[0], solver.state_vars[1]]
        placement = obstacles[lidar_config[i]][4]
        angle     = laser_angles[i]            
        # TODO: tan, cot do work for horizontal and vertical lasers.
        # TODO: Convert angles to radians out of loop.
        # TODO: Better way to compute cot, maybe numpy.
        if placement: # obstacle is vertical
            obst_x    = obstacles[lidar_config[i]][0]
            tan_angle = math.tan(math.radians(angle))
            A = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, tan_angle, 0.0]]
            #A = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, tan_angle, -1.0]]
            b = [obst_x, obst_x * tan_angle]

        else: # obstacle is horizontal
            obst_y    = obstacles[lidar_config[i]][1]
            cot_angle = math.cos(math.radians(angle)) / math.sin(math.radians(angle))
            A = [[1.0, 0.0, 0.0, cot_angle], [0.0, 1.0, 0.0, 1.0]]     
            #A = [[1.0, 0.0, -1.0, cot_angle], [0.0, 1.0, 0.0, 0.0]]     
            b = [obst_y * cot_angle, obst_y]

        solver.add_linear_constraints(A, rVars, b, sense=0)  #EQ constraint


def add_dynamics_constraints(solver):

    # x integrators
    # x+ = x- + vx*Ts
    rVars = [solver.next_state_vars[0], solver.state_vars[0],solver.out_vars[0]]
    #print 'chainX', rVars
    solver.add_linear_constraints([[1.0, -1.0, -1*Ts]], rVars, [0.0], sense= 0) #EQ constraint

    # y integrators
    rVars = [solver.next_state_vars[1], solver.state_vars[1],solver.out_vars[1]]

    #print 'chainY', rVars
    solver.add_linear_constraints([[1.0, -1.0, -1*Ts]], rVars, [0.0], sense= 0) #EQ constraint


    # # x last state, such as vx(t+1) = vx(t) + Ts * ux(t)
    # rVars = [solver.next_state_vars[2],solver.state_vars[2], solver.out_vars[0]]

    # #print 'chainX last', rVars          
    # solver.add_linear_constraints([[1.0, -1.0, -1*Ts]], rVars, [0.0], sense= 0)

    # # y last state, such as vy(t+1) = vy(t) + Ts * uy(t)
    # rVars = [solver.next_state_vars[3],solver.state_vars[3], solver.out_vars[1]]

    # #print 'chainY last', rVars          
    # solver.add_linear_constraints([[1.0, -1.0, -1*Ts]], rVars, [0.0], sense= 0)

def in_region(regions,x):
    
    ret = []
    for idx,region in enumerate(regions):
        H = np.array(region['A'])
        b = np.array(region['b'])
        diff = H.dot(x) -b
        if  (diff<=eps).all():
            print(idx) 
            ret.append(idx)
    return ret

def preprocess_regions(regions_H_rep, regions):
    bool_model = [False] * hidden_units
    for start_region in regions:
        solver = Solver(input_size,hidden_units,output_dim, network = weights)
        print('Preprocessing Network, Region %d'%start_region)
        s = time()
        add_initial_state_constraints(solver, regions_H_rep, start_region)
        add_lidar_constraints(solver, start_region)
        counter_example, bool_model = solver.solve()
        e = time()  
        print('time preprocessing', e-s)
        filename = 'Region_' + str(start_region)
        with open('counterexamples/'+filename,'wb') as f:
            pickle.dump({'counter_example':counter_example, 'bool_model': bool_model}, f)

def add_initial_partition_constraints(solver, partition):

    state_vars = [solver.state_vars[0],solver.state_vars[1]]
    A, b = partition['PolygonH']['A'], partition['PolygonH']['b']
    solver.add_linear_constraints(A, state_vars, b)

    # state_vars = [solver.state_vars[2],solver.state_vars[3]]        
    # A = [[1.0, 0.0],[0.0,1.0],[-1.0,0.0],[0.0,-1.0]]
    # v_upper_bound = partition['HyperCube']['max']
    # v_lower_bound = partition['HyperCube']['min']
    # b = [v_upper_bound[0],v_upper_bound[1], -1*v_lower_bound[0], -1*v_lower_bound[1]]
    # solver.add_linear_constraints(A, state_vars, b)



def add_final_partition_constraints(solver, partition):

    state_vars = [solver.next_state_vars[0],solver.next_state_vars[1]]
    A, b = partition['PolygonH']['A'], partition['PolygonH']['b']
    solver.add_linear_constraints(A, state_vars, b)

    # state_vars = [solver.state_vars[2],solver.state_vars[3]]        
    # A = [[1.0, 0.0],[0.0,1.0],[-1.0,0.0],[0.0,-1.0]]
    # v_upper_bound = partition['HyperCube']['max']
    # v_lower_bound = partition['HyperCube']['min']
    # b = [v_upper_bound[0],v_upper_bound[1], -1*v_lower_bound[0], -1*v_lower_bound[1]]
    # solver.add_linear_constraints(A, state_vars, b)



def add_preprocessing_tasks(partitions, task_Q):
    for partition in partitions:
        if(partition['isObstacle']):
            continue
        task_Q.put(partition)
    

        

def preprocess_partition(args):
    task_Q, print_lock, file_lock,additional_ce,process_num = args
    try:
        while(not task_Q.empty()):
            partition = task_Q.get_nowait()
            state_idx = partition['SymbolicStateIndex']
            solver = Solver(input_size,hidden_units,output_dim, network = weights)
            print_lock.acquire()
            print('Preprocessing Network, state %d'%state_idx,'process:', process_num)
            print_lock.release()
            region_idx = partition['RegionIndex']

            #Load Region counter examples
            filename = 'Region_' + str(region_idx)
            s = time()
            f =  open('counterexamples/'+filename,'rb')
            dict = pickle.load(f)
            counter_example, bool_model = dict['counter_example'], dict['bool_model']
            #Add Region counter examples
            solver.add_counter_example(counter_example,bool_model, AND = True)
            #continue preprocessing region
            add_initial_partition_constraints(solver, partition)
            add_lidar_constraints(solver, region_idx)
            pre_counter_example, bool_model = solver.solve(bool_model)
            diffLenCE = len(pre_counter_example) - len(counter_example)
            if(diffLenCE):
                additional_ce.value += 1
            e = time()  
            print_lock.acquire()
            print('time preprocessing', e-s)
            print_lock.release()
            filename = 'state_' + str(state_idx)
            with open('counterexamples/'+filename,'wb') as f:
                pickle.dump({'counter_example':pre_counter_example, 'bool_model': bool_model, 'diff':diffLenCE}, f)
    except Exception as e:
        print(e)
        return

def solveLP(args):
    try:
        task_Q, weights, unsafe, log_file, process_num, print_lock,log_file_lock,running_jobs, graph = args
        
        while(True):
            if(task_Q.empty() and running_jobs.value == 0):
                return
            elif(task_Q.empty()):
                sleep(5)
                print('process sleeping',process_num,running_jobs.value)
                continue

            start_partition, end_partition  = task_Q.get_nowait()
            end_partition_idx   = end_partition['SymbolicStateIndex']
            start_partition_idx  = start_partition['SymbolicStateIndex']
            start_region = start_partition['RegionIndex']

            if(start_partition_idx in unsafe):
                continue

            print_lock.acquire()
            running_jobs.value += 1
            print_lock.release()
            print_lock.acquire()
            print('Start state:',start_partition_idx,'End state:', end_partition_idx, 'process:', process_num)
            print_lock.release()
            solver = Solver(input_size,hidden_units,output_dim,network = weights, process_num = process_num, print_lock = print_lock)
            solver.preprocessing = False
            #load counter examples

            filename = 'state_' + str(start_partition_idx)
            with open('counterexamples/'+filename,'rb') as f:
                data = pickle.load(f)
                counter_example, bool_model = data['counter_example'], data['bool_model']
            solver.add_counter_example(counter_example,bool_model,AND=True)
            # print('Solving with Output and Dynamics constraints')
            s = time()
            add_initial_partition_constraints(solver, start_partition)
            add_lidar_constraints(solver, start_region)
            add_final_partition_constraints(solver,end_partition)
            add_dynamics_constraints(solver)
            vars,counter_examples,status = solver.solve()
            e = time()
            log_file_lock.acquire()
            f= open(log_file,'a')
            f.write(str(start_partition_idx)+'-->'+str(end_partition_idx)+':')
            f.write(status + ', time: ' + str(e-s) + '\n')
            f.close()
            log_file_lock.release()
            graph[start_partition_idx][end_partition_idx] = {'status':status, 'vars':vars}
            if(status == 'SolFound' and start_partition_idx not in unsafe):
                unsafe.append(start_partition_idx)
                # add_tasks(task_Q,start_partition_idx)
            print_lock.acquire()
            running_jobs.value -=1
            print_lock.release()
            
        
    except Exception as e:
        print(e)
        return
            
def add_tasks(task_Q, end_partition_idx):
    end_partition = partitioner.symbolic_states[end_partition_idx]
    adjacents = set(end_partition['Adjacents'])
    for _DICT_TYPE,start_partition_idx in enumerate(adjacents):
        start_partition = partitioner.symbolic_states[start_partition_idx]
        if(start_partition['isObstacle']):
            continue
        print(start_partition_idx, end_partition_idx)
        task_Q.put((start_partition,end_partition))

if __name__ == "__main__":
    
    abst_reg_H_rep, lidar_config_dict = load_regions()
    num_obstacles = 3
    # in_region(abst_reg_H_rep,np.array([5.7,2]))
    num_lasers = 8
    workspace = Workspace(8,num_lasers,'obstacles.json')
    obstacles    = workspace.lines
    laser_angles     = workspace.laser_angles        
    num_integrators = 1
    Ts = 0.05
    u_bound = 0.5

    StateSpacePartioner_DEBUG = False # local debug flag
    higher_deriv_bound = 1.0
    grid_size = [0.05,1.0]
    neighborhood_radius = 0.01
    dir = 'NN20_grid_005/'
    
    models_filenames = glob.glob('models/model*')
    models_filenames.sort()
    LOAD = True
    if(LOAD):
        with open(dir +'symbolic_states','rb') as f:
            partitioner = pickle.load(f)
    else:
        partitioner = StateSpacePartitioner(workspace, num_integrators, higher_deriv_bound, grid_size, neighborhood_radius)
        partitioner.partition()
    with open(dir + 'symbolic_states','wb') as f:
        pickle.dump(partitioner,f)
        f.close()
    for filename in models_filenames: #For each Neural network
        network = load_model(filename)
        print('Loaded network from %s' %filename)
        words = filename.split('_')
        epoch = words[-1].split('.')[0]
        if(int(epoch) < 30):
            continue
        hidden_weights= network.layers[0].get_weights()
        output_weights= network.layers[-1].get_weights()
        weights = [hidden_weights, output_weights]
        input_size = 16
        output_dim = 2
        hidden_units = 20
        # with open(weights_file,'rb') as f:
        #     weights = pickle.load(f)
        preprocessRegions = False
        preprocessPartitions = False
        if(preprocessRegions):
            preprocess_regions(partitioner.regions, range(len(partitioner.regions) - num_obstacles))


        
        if(preprocessPartitions):
            m = Manager()
            task_Q = m.Queue()
            print_lock = m.Lock()
            file_lock = m.Lock()
            additional_ce = m.Value('i',0)
            num_workers = 8
            process_num = 0
            pool = Pool(num_workers)
            add_preprocessing_tasks(partitioner.symbolic_states, task_Q)
            for i in range(num_workers):
                args=[task_Q, print_lock, file_lock,additional_ce,process_num]
                pool.apply_async(preprocess_partition,(args,))
                process_num +=1
            try:
                pool.close()
                pool.join()
                print('Number of states with additional CE: %d'%additional_ce.value)
            except Exception as e:
                print(e)
        PARALLEL = True

        # with open('safe_points','rb') as f:
        # #     start_points = pickle.load(f)
        #     partitioner.plotWorkspacePartitions([6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 144, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 250, 251, 252, 253, 254, 257, 258, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 336, 342, 343, 344, 345, 346, 347, 348, 351, 352, 360, 362, 366, 367, 371, 372, 373, 377, 378, 379, 380, 384, 385, 386, 387, 388, 392, 393, 394, 395, 396, 397, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 432, 440, 441, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498])
        #     # partitioner.no_partition()


        f_result_name = 'results/results_%s.txt'%epoch
        open(f_result_name, 'w').close()
        if(PARALLEL):
            m = Manager()
            unsafe = m.list()
            task_Q = m.Queue()
            print_lock = m.Lock()
            log_file_lock = m.Lock()
            running_jobs = m.Value('i',0)
            process_num = 0
            graph = m.dict()
            for i in range(len(partitioner.symbolic_states)):
                graph[i] = {}
            num_workers = 7
            pool = Pool(num_workers)
            results = []
            for i in range(num_workers):
                args=[task_Q, weights, unsafe, f_result_name, process_num,print_lock,log_file_lock, running_jobs,graph]
                results.append(pool.apply_async(solveLP,(args,)))
                process_num +=1
            print_lock.acquire()
            for obstacle_state_index in set(partitioner.obstacle_symbolic_states):
                add_tasks(task_Q, obstacle_state_index)
                unsafe.append(obstacle_state_index)
            print_lock.release()
            
            s = time()
            try:
                pool.close()
                # pool.join()
                for r in results:
                    r.get()
                e = time()
                log_file= open(f_result_name,'a')
                log_file.write("============ Time: " + str(e - s) + " ============\n")
                log_file.close()
                with open(dir +'graph_%s'%epoch,'wb') as f:
                    pickle.dump(graph._getvalue(),f)
                    f.close()
                with open(dir + 'safety_%s'%epoch,'wb') as f:
                    pickle.dump(unsafe._getvalue(),f)
                    f.close()
            except KeyboardInterrupt as e:
                print(e)
                with open(dir +'graph_%s'%epoch,'wb') as f:
                    pickle.dump(graph._getvalue(),f)
                with open(dir + 'safety_%s'%epoch,'wb') as f:
                    pickle.dump(unsafe._getvalue(),f)

                print(running_jobs.value)
            except Exception as e:
                print(e)
        
        
        else:
            pass
    