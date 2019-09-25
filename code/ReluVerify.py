from solver import *
from time import time
from random import random, seed
import numpy as np
from keras.models import load_model


def load_regions(dir = "./regions/"):
    filename = dir + 'abst_reg_H_rep.txt'
    with open(filename, 'rb') as inputFile:
        abst_reg_H_rep = pickle.load(inputFile)

    
    filename = dir + 'lidar_config_dict.txt'
    with open(filename, 'rb') as inputFile:
        lidar_config_dict = pickle.load(inputFile)
    return abst_reg_H_rep, lidar_config_dict

def add_initial_state_constraints(solver, start_region):

    state_vars = [solver.state_vars[0],solver.state_vars[1]]
    region  = abst_reg_H_rep[start_region]
    A, b = region['A'], region['b']
    solver.add_linear_constraints(A, state_vars, b)

    state_vars = [solver.state_vars[2],solver.state_vars[3]]
    A = [[1.0, 0.0],[0.0,1.0],[-1.0,0.0],[0.0,-1.0]]
    b = [u_bound] * 4
    solver.add_linear_constraints(A, state_vars, b)

    # A = [[1.0, 0.0], [-1.0,0], [0,1.0],[0,-1.0]]
    # b = [6, 0, 6, 0.0]
    # solver.add_linear_constraints(A, [solver.state_vars[0],solver.state_vars[1]], b)
    
    # A = [[1.0, 0.0,0.0,0.0], [0,1,0,0], [0,0,1,0],[0,0,0,1]]
    # b = [0.9838070389814675, 2.425289113074541, 0.10965773928910494, 0.29743393510580063]
    # solver.add_linear_constraints(A, solver.state_vars, b,sense = 0)

def add_final_state_constraints(solver, end_region):
    state_vars = [solver.next_state_vars[0],solver.next_state_vars[1]]
    region  = abst_reg_H_rep[end_region]
    A, b = region['A'], region['b']
    solver.add_linear_constraints(A, state_vars, b)

    state_vars = [solver.next_state_vars[2],solver.next_state_vars[3]]
    A = [[1.0, 0.0],[0.0,1.0],[-1.0,0.0],[0.0,-1.0]]
    b = [u_bound] * 4
    solver.add_linear_constraints(A, state_vars, b)

    # A = [[1.0, 0.0,0.0,0.0], [0,1,0,0], [0,0,1,0],[0,0,0,1]]
    # b = [1.03863590862602, 2.5740060806274414, 0.13783891778439283, 0.26302529126405716]
    # solver.add_linear_constraints(A, solver.next_state_vars, b,sense = 0)

def add_lidar_constraints(start_region):
    """
    For a certain laser i, if it intersects a vertical obstacle:
        x_i = x_obstacle
        y_i = y_car + (x_obstacle - x_car) tan(laser_angle)
    Otherwise:
        x_i = x_car + (y_obstacle - y_car) cot(laser_angle)
        y_i = y_obstacle
    """


    # A = np.eye(16)
    # b = [1.959213450551033, 0.24507476051007138, -0.2450747605100717, -0.8972459780052304, -0.8972459780052304, -0.1927344335540081, 0.19273443355400754, 1.540786549448966, 1.959213450551033, 1.959213450551033, 1.959213450551033, 0.8972459780052309, -0.8972459780052302, -1.540786549448967, -1.540786549448967, -1.540786549448967]
    # solver.add_linear_constraints(A, solver.im_vars, b, sense=0)  #EQ constraint
    

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
    rVars = [solver.next_state_vars[0], solver.state_vars[0],solver.state_vars[2]]
    #print 'chainX', rVars
    solver.add_linear_constraints([[1.0, -1.0, -1*Ts]], rVars, [0.0], sense= 0) #EQ constraint

    # y integrators
    rVars = [solver.next_state_vars[1], solver.state_vars[1],solver.state_vars[3]]

    #print 'chainY', rVars
    solver.add_linear_constraints([[1.0, -1.0, -1*Ts]], rVars, [0.0], sense= 0) #EQ constraint


    # x last state, such as vx(t+1) = vx(t) + Ts * ux(t)
    rVars = [solver.next_state_vars[2],solver.state_vars[2], solver.out_vars[0]]

    #print 'chainX last', rVars          
    solver.add_linear_constraints([[1.0, -1.0, -1*Ts]], rVars, [0.0], sense= 0)

    # y last state, such as vy(t+1) = vy(t) + Ts * uy(t)
    rVars = [solver.next_state_vars[3],solver.state_vars[3], solver.out_vars[1]]

    #print 'chainY last', rVars          
    solver.add_linear_constraints([[1.0, -1.0, -1*Ts]], rVars, [0.0], sense= 0)

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

if __name__ == "__main__":

    
    abst_reg_H_rep, lidar_config_dict = load_regions()
    start_region = 2
    end_region = 4
    in_region(abst_reg_H_rep,np.array([1.03863590862602, 2.5740060806274414]))
    num_lasers = 8
    workspace = Workspace(8,num_lasers,'obstacles.json')
    obstacles    = workspace.lines
    laser_angles     = workspace.laser_angles        
    num_integrators = 2
    Ts = 0.5
    u_bound = 0.5

    
    input_size = 16
    output_dim = 2
    hidden_units = 600
    model = load_model('my_model.h5')
    solver = Solver(input_size,hidden_units,output_dim, hidden_units, network = model)
    
    
    add_initial_state_constraints(solver, start_region)
    add_lidar_constraints(start_region)
    print('Preprocessing Network')
    s = time()
    vars,counter_examples = solver.solve()
    e = time()  
    print('time preprocessing', e-s)
    print('Done Preprocessing')
    
    print('Solving with Output and Dynamics constraints')
    s = time()
    add_final_state_constraints(solver,end_region)
    add_dynamics_constraints(solver)
    vars,counter_examples = solver.solve()
    e = time()  
    print('time', e-s)