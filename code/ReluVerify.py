from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
# from keras.models import load_model
from StateSpacePartitioner import StateSpacePartitioner
import pandas as pd
import sys,os
from multiprocessing import Queue,Lock,Pool,current_process,Manager
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
    A, b = region_H['A'], region_H['b']
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
    
    ret = -1
    for idx,region in enumerate(regions):
        H = np.array(region['A'])
        b = np.array(region['b'])
        diff = H.dot(x) -b
        if  (diff<=eps).all():
            print(idx)
            if(ret != -1):
              return -2
            else:
                ret = idx
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
    while(not task_Q.empty()):
        partition = task_Q.get()
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

def  solveLP(args):
    try:
        task_Q, weights, unsafe, log_file, process_num, print_lock,log_file_lock,running_jobs, graph = args
        
        while(True):
            if(task_Q.empty() and running_jobs.value == 0):
                return
            elif(task_Q.empty()):
                sleep(5)
                continue

            start_partition, end_partition  = task_Q.get()
            end_partition_idx   = end_partition['SymbolicStateIndex']
            start_partition_idx  = start_partition['SymbolicStateIndex']
            start_region = start_partition['RegionIndex']

            if(start_partition_idx in unsafe):
                continue

            running_jobs.value += 1
            print_lock.acquire()
            print('Start state:',start_partition_idx,'End state:', end_partition_idx, 'process:', process_num)
            print_lock.release()
            solver = Solver(network = weights, process_num = process_num, print_lock = print_lock)
            solver.preprocessing = False
            #load counter examples

            filename = 'state_' + str(start_partition_idx)
            log_file_lock.acquire()
            with open('counterexamples/'+filename,'rb') as f:
                data = pickle.load(f)
                log_file_lock.release()
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
            graph[start_partition_idx][end_partition_idx] = status
            if(status == 'SolFound' and start_partition_idx not in unsafe):
                unsafe.append(start_partition_idx)
                add_tasks(task_Q,start_partition_idx)
            running_jobs.value -=1
            
        
    except KeyboardInterrupt as e:
        raise KeyboardInterruptError()
            
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
    start_partition_idx = 2
    end_region = 4
    # in_region(abst_reg_H_rep,np.array([5.7,2]))
    num_lasers = 8
    workspace = Workspace(8,num_lasers,'obstacles.json')
    obstacles    = workspace.lines
    laser_angles     = workspace.laser_angles        
    num_integrators = 1
    Ts = 0.5
    u_bound = 0.5

    StateSpacePartioner_DEBUG = False # local debug flag
    higher_deriv_bound = 1.0
    grid_size = [0.25,1.0]
    neighborhood_radius = 0.1
    weights_file = 'weights'


    input_size = 16
    output_dim = 2
    hidden_units = 600
    with open(weights_file,'rb') as f:
        weights = pickle.load(f)
    preprocessRegions = False
    preprocessPartitions = True
    if(preprocessRegions):
        preprocess_regions(abst_reg_H_rep, len(abst_reg_H_rep) - num_obstacles)
        sys.exit()

    partitioner = StateSpacePartitioner(workspace, num_integrators, higher_deriv_bound, grid_size, neighborhood_radius)
    partitioner.partition()
    
    if(preprocessPartitions):
        m = Manager()
        task_Q = m.Queue()
        print_lock = m.Lock()
        file_lock = m.Lock()
        additional_ce = m.Value('i',0)
        num_workers = 7
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
        sys.exit()
    PARALLEL = True

    # with open('safe_points','rb') as f:
    #     start_points = pickle.load(f)
    #     partitioner.plotWorkspacePartitions([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 149, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 299, 300, 301, 302, 304, 305, 307, 308, 309, 312, 313, 314, 315, 316, 353, 354, 355, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 466, 467, 468, 469, 470])
    #     # partitioner.no_partition()


    f_result_name = 'results.txt' 
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
        for i in range(num_workers):
            args=[task_Q, weights, unsafe, f_result_name, process_num,print_lock,log_file_lock, running_jobs,graph]
            pool.apply_async(solveLP,(args,))
            process_num +=1
        print_lock.acquire()
        for obstacle_state_index in set(partitioner.obstacle_symbolic_states):
            add_tasks(task_Q, obstacle_state_index)
            unsafe.append(obstacle_state_index)

        print_lock.release()
        
        s = time()
        try:
            pool.close()
            pool.join()
            e = time()
            log_file= open(f_result_name,'a')
            log_file.write("============ Time: " + str(e - s) + " ============\n")
            log_file.close()
            with open('graph','wb') as f:
                pickle.dump(graph._getvalue(),f)
            with open('safety','wb') as f:
                pickle.dump(unsafe._getvalue(),f)
        except KeyboardInterrupt as e:
            print(e)
            with open('graph','wb') as f:
                pickle.dump(dict(graph),f)

       
    
    else:
        pass
    
    