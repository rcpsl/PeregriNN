## This file needs the following packages
# 1- ShapeLY
# 2- PycddLib. This package needs GMP (Gnu Multi Prescision) https://brewinstall.org/Install-gmp-on-Mac-with-Brew/

import pickle
from shapely.geometry import Polygon
from shapely.geometry import box

import numpy as np
import heapq as heap

import cdd

from matplotlib import pyplot as plt
import math

from Workspace import Workspace


StateSpacePartioner_DEBUG = False

# Terminology:
# Workspace = 2D workspace
# Statespace = the whole n-dimensional state space of the robot
# Region = subset of the workspace
# Partition = subset of the statespace

class StateSpacePartitioner(object):
   
    ################################################################################################
    #
    #       Constructor
    #
    ################################################################################################
    def __init__(self, workspace, num_integrators, higher_deriv_bound, grid_size, neighborhood_radius):

        self.workspace = {'Polygon':box(0,0,6,6), 'Obstacles':[]}

        self.num_integrators = num_integrators          # controls the dimension of the state space

        self.higher_deriv_bound = higher_deriv_bound    # the norm on higher order derivatives (used to compute the
                                                        # number of partitions

        self.position_grid_size = grid_size[0]                    # the grid size used to partition the state space
        self.high_order_deriv_grid_size = grid_size[1]

        self.neighborhood_radius = neighborhood_radius  # controls how far we consider two partitions to be adjacent
                                                        # i.e., two partitions X_1 and X_2 are said to be adjacent
                                                        # if there exists x_1 \in X_1 and x_2 in X_2 such that
                                                        # distance(x_1,x_2) < neighborhood_radius
        self.regions    = []         # data structure that holds the information for all workspace partitions
        self.cubes      = []         # data structure that holds the information for hyper cubes in the higher-order dimensions

        self.symbolic_states = []   # data structure that holds the information for all symbolic states (state space partitions)
        self.obstacle_symbolic_states = [] #indecies of all obstacle symbolic states

        self.__RegionPartitionIndexShift = []
        self.__NumberOfRegionPartitions = 0

    ################################################################################################
    #
    #       Partition
    #
    ################################################################################################
    def partition(self):
        self.__partition_wrokspace()


        self.__partition_higher_order_derivatives()


        self.__partition_statespace()

        # self.__plotWorkspaceRegions()
        # self.plotWorkspacePartitions()
        x = 0


    def no_partition(self):
        self.__partition_wrokspace()
        self.obstacle_symbolic_states = []
        for region_idx,region in enumerate(self.regions):
            region['SymbolicStateIndex'] = region_idx
            region['RegionIndex'] = region_idx
           
            if(region['isObstacle']):
                self.obstacle_symbolic_states.append(region_idx)
        self.symbolic_states = self.regions
        


    ################################################################################################
    #
    #       Partition Workspace:
    #           private function that takes workspace as input and partition it into regions
    #           based on the LiDAR angles.
    #           It creates the "regions" data structure which contains:
    #           1- 'Polygon': The polygon information of each region (using ShapeLy polygon objects)
    #           2- 'Adjacents': a list of adjacent workspace regions. Two regions are considered
    #                           adjacent if they share at least one point on their boundaries
    #           3- 'isObstacle': A flag on whether the region is an obstacle
    #
    ################################################################################################
    def __partition_wrokspace(self):
        #----------------------------------------------------------------------------
        #   STEP 1: Partition the workspace
        #----------------------------------------------------------------------------
        # TBD: re-implement the workspace partitioning here using Shapely package
        #      for now, we will import the regions from files

        num_obstacles = 3
        filename = './regions/abst_reg_H_rep.txt'
        with open(filename, 'rb') as inputFile:
            abst_reg_H_rep = pickle.load(inputFile)
            A = [[0.0, -1.00000000e+00],
                [-1.00000000e+00, -0.00000000e+00],
                [ 0.0,  1.00000000e+00],
                [ 1.00000000e+00, -0.00000000e+00]]
            b = [[0.0],
                [-2.50000000e+00],
                [2.5],
                [ 6.00000000e+00]]
            abst_reg_H_rep[56]['A'] = A
            abst_reg_H_rep[56]['b'] = b
            A = [[0.0, -1.00000000e+00],
                [-1.00000000e+00, -0.00000000e+00],
                [ 0.0,  1.00000000e+00],
                [ 1.00000000e+00, -0.00000000e+00]]
            b = [[-2.5],
                [-5.50000000e+00],
                [6.0],
                [ 6.00000000e+00]]
            abst_reg_H_rep.append({'A':A, 'b':b})
        filename = './regions/abst_reg_V_rep.txt'
        with open(filename, 'rb') as inputFile:
            abst_reg_V_rep = pickle.load(inputFile)
            v = [(2.5, 0), (2.5, 2.5), (5.5, 2.5), (5.5, 0)]
            abst_reg_V_rep[56] = v
            v = [(5.5, 2.5), (5.5, 6.0), (6.0, 6.0), (6.0, 2.5)]
            abst_reg_V_rep.append(v)



        numberOfWorkspaceRegions = len(abst_reg_H_rep) - num_obstacles
        for index in range(numberOfWorkspaceRegions):
            self.__RegionPartitionIndexShift.append(self.__NumberOfRegionPartitions)
            region = Polygon(abst_reg_V_rep[index])
            A, b = self.__computeHRepresentation(region)
            self.regions.append({'Polygon': region,
                                 'PolygonH': {'A': A, 'b': b},
                                 'Adjacents': [],
                                 'isObstacle': False,
                                 'Partitions': []
                                 })

            # re-partition the region using the gird_size parameter
            region_partitions = self.__partition_region(index)
            # for idx,partition in enumerate(region_partitions):
            #     if(partition['Polygon'].area < 1E-10):
            #         region_partitions.pop(idx)
            self.regions[index]['Partitions'] = region_partitions

            self.__NumberOfRegionPartitions = self.__NumberOfRegionPartitions + len(region_partitions)



        # TBD: the loaded files (above) do not include workspace obstacles, add them manually for now
        obstacles_idx = [55,56,57]
        # obstacle = Polygon([(2.5,0), (5.5, 0), (5.5, 2.5), (2.5, 2.5)])
        for obs_idx in obstacles_idx:
            obstacle = Polygon(abst_reg_V_rep[obs_idx])
            A, b = self.__computeHRepresentation(obstacle)
            self.regions.append({'Polygon': obstacle,
                                'PolygonH': {'A': A, 'b': b},
                                'Adjacents': [],
                                'isObstacle': True,
                                # do not re-partition obstacles. It has only one partition = obstacle
                                'Partitions': [{'Polygon': obstacle,
                                                'PolygonH': {'A': A, 'b': b},
                                                'RegionIndex': len(self.regions),
                                                'PartitionIndex': 0,
                                                'Adjacents': [],
                                                'isObstacle': True
                                                }]
                                })
            self.__RegionPartitionIndexShift.append(self.__NumberOfRegionPartitions)
            self.__NumberOfRegionPartitions = self.__NumberOfRegionPartitions + 1

        # obstacle = Polygon([(0, 3.5), (0,6), (4, 6), (4, 3.5)])
        # A, b = self.__computeHRepresentation(obstacle)
        # self.regions.append({'Polygon': obstacle,
        #                      'PolygonH': {'A': A, 'b': b},
        #                      'Adjacents': [],
        #                      'isObstacle': True,
        #                      # do not re-partition obstacles. It has only one partition = obstacle
        #                      'Partitions': [{'Polygon': obstacle,
        #                                       'PolygonH': {'A': A, 'b': b},
        #                                       'RegionIndex': len(self.regions),
        #                                       'PartitionIndex': 0,
        #                                       'Adjacents': [],
        #                                       'isObstacle': True
        #                                     }]
        #                      })
        # self.__RegionPartitionIndexShift.append(self.__NumberOfRegionPartitions)
        # self.__NumberOfRegionPartitions = self.__NumberOfRegionPartitions + 1
        # ----------------------------------------------------------------------------
        #   STEP 2: Check for Adjacent Workspace Regions
        # ----------------------------------------------------------------------------
        for index1 in range(len(self.regions)):
            self.regions[index1]['Adjacents'].append(index1)
            for index2 in range(index1+1, len(self.regions)):
                polygon1 = self.regions[index1]['Polygon']
                polygon2 = self.regions[index2]['Polygon']

                distance = polygon1.distance(polygon2)
                if distance < self.neighborhood_radius:
                    self.regions[index1]['Adjacents'].append(index2)
                    self.regions[index2]['Adjacents'].append(index1)

        # ----------------------------------------------------------------------------
        #   STEP 3: Check for Adjacent Workspace Partitions
        # ----------------------------------------------------------------------------
        # only check those partitions in the regions that are adjacent to the current region
        for region_index in range(len(self.regions)):
            for partition in self.regions[region_index]['Partitions']:
                polygon1 = partition['Polygon']
                for other_region_index in self.regions[region_index]['Adjacents']:
                # adjacent_regions = self.regions[region_index]['Adjacents']
                # while adjacent_regions:
                #     dist, other_region_index = heap.heappop(adjacent_regions) #this will return a pair of distance
                #                                                               # and region index
                    for other_partition in self.regions[other_region_index]['Partitions']:
                        if(partition['PartitionIndex'] == other_partition['PartitionIndex'] and region_index == other_region_index):
                            continue
                        polygon2 = other_partition['Polygon']

                        distance = polygon1.distance(polygon2)
                        if distance < self.neighborhood_radius:
                            partition['Adjacents'].append({
                                            'RegionIndex': other_partition['RegionIndex'],
                                            'PartitionIndex': other_partition['PartitionIndex']}
                                          )
                            other_partition['Adjacents'].append({
                                            'RegionIndex': partition['RegionIndex'],
                                            'PartitionIndex': partition['PartitionIndex']}
                                          )

        # ----------------------------------------------------------------------------
        #   STEP 4: Plot all regions and their adjacents
        # ----------------------------------------------------------------------------
        if StateSpacePartioner_DEBUG:
            # Just for debugging: for each region, plot the adjacent ones
            for index1 in range(len(self.regions)):
                polygon = self.regions[index1]['Polygon']
                fig = plt.figure(1, figsize=(5, 5), dpi=90)
                axis = fig.add_subplot(111)
                axis.set_title("Adjacents of region " + str(index1))
                self.__plotWorkspacePolygon(polygon, str(index1), axis)

                adjacents = self.regions[index1]['Adjacents']
                for index2 in range(len(adjacents)):
                    region_index = self.regions[index1]['Adjacents'][index2]
                    polygon2 = self.regions[region_index]['Polygon']
                    self.__plotWorkspacePolygon(polygon2, str(region_index), axis)

                plt.show()


        # ----------------------------------------------------------------------------
        #   STEP 5: Plot all partitions and their adjacents
        # ----------------------------------------------------------------------------
        if StateSpacePartioner_DEBUG:
            # Just for debugging: for each region, plot the adjacent ones
            for region_index in range(len(self.regions)):
                partitions = self.regions[region_index]['Partitions']
                for partition_index in range(len(partitions)):
                    partition = partitions[partition_index]
                    polygon = partition['Polygon']
                    fig = plt.figure(1, figsize=(5, 5), dpi=90)
                    axis = fig.add_subplot(111)
                    axis.set_title("Adjacents of Partition " + str(partition_index) + " in Region " + str(region_index))
                    self.__plotWorkspacePolygon(polygon, str(region_index)+":"+str(partition_index), axis)

                    adjacents = partition['Adjacents']
                    for adjacent_index in range(len(adjacents)):
                        adjacent_region_index = adjacents[adjacent_index]['RegionIndex']
                        adjacent_partition_index = adjacents[adjacent_index]['PartitionIndex']
                        polygon2 = self.regions[adjacent_region_index]['Partitions'][adjacent_partition_index]['Polygon']
                        self.__plotWorkspacePolygon(polygon2, str(adjacent_region_index) + ":" + str(adjacent_partition_index), axis)

                plt.show()



    #------------------------------------------------------------------------------
    # Helper Function:
    #               Repartition a workspace region based on the grid_size parameter
    # ------------------------------------------------------------------------------
    def __partition_region(self, region_index):
        region_partitions = []

        region = self.regions[region_index]
        x_min = region['Polygon'].bounds[0]
        y_min = region['Polygon'].bounds[1]
        x_max = region['Polygon'].bounds[2]
        y_max = region['Polygon'].bounds[3]
        
        number_of_partitions_x = int(math.ceil((x_max - x_min) / self.position_grid_size))
        number_of_partitions_y = int(math.ceil((y_max - y_min) / self.position_grid_size))

        for index_x in range(number_of_partitions_x):
            for index_y in range(number_of_partitions_y):
                x_start_partition = x_min + index_x * self.position_grid_size
                x_end_partition = x_start_partition + self.position_grid_size

                y_start_partition = y_min + index_y * self.position_grid_size
                y_end_partition = y_start_partition + self.position_grid_size

                cell = box(x_start_partition, y_start_partition, x_end_partition, y_end_partition)

                # compute the intersection between the cell and the region
                partition_polygon = region['Polygon'].intersection(cell)

                # check if the intersection is not just a point or a line, but a polygon
                if partition_polygon.geom_type == 'Polygon':
                    if(partition_polygon.area < 1E-10):
                        continue
                    A, b = self.__computeHRepresentation(partition_polygon)

                    # Add the partition to the dictionary, do not worry about the hypercubes and the adjacents for now
                    region_partitions.append({'Polygon': partition_polygon,
                                              'PolygonH': {'A': A, 'b': b},
                                              'RegionIndex': region_index,
                                              'PartitionIndex': len(region_partitions),
                                              'Adjacents': [],
                                              'isObstacle': False
                                              })

        # add the adjacent regions (those only within the same workspace region)
        for index_partition_1 in range(len(region_partitions)):
            for index_partition_2 in range(index_partition_1 + 1, len(region_partitions)):
                distance = region_partitions[index_partition_1]['Polygon'].distance(
                    region_partitions[index_partition_2]['Polygon'])
                if distance < self.neighborhood_radius:
                    region_partitions[index_partition_1]['Adjacents'].append(
                                  {'RegionIndex': region_index, 'PartitionIndex': index_partition_2})

                    region_partitions[index_partition_2]['Adjacents'].append(
                                  {'RegionIndex': region_index, 'PartitionIndex': index_partition_1})

        # Just for debugging, plot the intersection between the cell and the region
        if StateSpacePartioner_DEBUG:
            fig = plt.figure(1, figsize=(5, 5), dpi=90)
            axis = fig.add_subplot(111)
            axis.set_title("Cell partitions of region " + str(region_index))
            self.__plotWorkspacePolygon(region['Polygon'], '', axis)

            for index1 in range(len(region_partitions)):
                print(region_partitions[index1]['Adjacents'])
                partition = region_partitions[index1]['Polygon']
                self.__plotWorkspacePolygon(partition, str(index1), axis)

            plt.show()

        return region_partitions
    ################################################################################################
    #
    #       Partition State Space:
    #           private function that creates a set of hypercubes using the
    #           "grid_size" parameter. This function creates the "cubes" data structure which
    #           contains the following information:
    #               1- 'HyperCube': A dictionary for the hypercube bounds of the partition
    #                   1.1- 'Min': A n-2 dimensional array that contains the min values of the cube in each dimension
    #                   1.2- 'Max': A n-2 dimensional array that contains the max values of the cube in each dimension
    #               2- 'Adjacents': A list of adjacents. Two partitions X_1 and X_2 are said to be adjacent
    #                               if there exists x_1 \in X_1 and x_2 in X_2 such that
    #                               distance(x_1,x_2) < neighborhood_radius
    #
    ################################################################################################
    def __partition_higher_order_derivatives(self):

        if self.num_integrators <= 1: #for 1-integrator, no higher order derivatives
            return

        grid_points = []
        points = np.linspace(-1 * self.higher_deriv_bound, self.higher_deriv_bound,
                             1 + np.floor((2 * self.higher_deriv_bound) / self.high_order_deriv_grid_size))
        for index in range(2 * (self.num_integrators-1)): #remove the x-y integrator
            grid_points.append(points[0:-1])    # we need the starting point of each cell,
                                                # so remove the very last end point

        grid = np.meshgrid(*grid_points)
        cell_positions = np.transpose(np.vstack(map(np.ravel, grid)))  # all points on the meshgrid

        for cell_start_position in cell_positions:
            hyper_cube_min = cell_start_position
            hyper_cube_max = cell_start_position + self.high_order_deriv_grid_size
            self.cubes.append({
                    'HyperCube': {'min': hyper_cube_min, 'max': hyper_cube_max},
                    'Adjacents': [],
                    })

        for cube_index in range(len(self.cubes)):
            hyper_cube_min = self.cubes[cube_index]['HyperCube']['min']
            adjacent_grid_points = []
            for dimension_index in range(len(hyper_cube_min)):
                start_point = hyper_cube_min[dimension_index]
                number_of_cubes = int(math.ceil(self.neighborhood_radius/self.high_order_deriv_grid_size))
                points = [start_point+i*self.high_order_deriv_grid_size for i in range(-number_of_cubes-1, number_of_cubes+2)]
                adjacent_grid_points.append(points)

            adjacent_grid = np.meshgrid(*adjacent_grid_points)
            adjacent_cell_positions = np.transpose(np.vstack(map(np.ravel, adjacent_grid))) # all points on the meshgrid


            for adjacent_cell_start in adjacent_cell_positions:
                if not np.all(adjacent_cell_start == hyper_cube_min):
                    # search for the row index for which cell_positions==adjacent_cell_start
                    cell_index = np.where(np.all(cell_positions==adjacent_cell_start, axis = 1))[0]
                    if cell_index: #if the cell we are looking for do exist (not outside the origional mesh grid)
                        self.cubes[cube_index]['Adjacents'].append(cell_index[0])

    ################################################################################################
    #
    #       Partition State Space:
    #           private function that takes the partioned workspace regions and further partitions
    #           it into small hypercubes (along with the higher order derivatives) using the
    #           "grid_size" parameter. This function creates the "partitions" data structure which
    #           contains the following information:
    #               1- 'Polygon': the polygon region of the partition
    #               2- 'PolygonH': A dictionary for the H-representation (half space representation) of the polygon Ax<b
    #                   2.1- 'A': the matrix part of the H-representation
    #                   2.2- 'b': the vector part of the H-representation
    #               3- 'HyperCube': A dictionary for the hypercube bounds of the partition
    #                   3.1- 'Min': A n dimensional array that contains the min values of the cube in each dimension
    #                   3.2- 'Max': A n dimensional array that contains the max values of the cube in each dimension
    #               4- 'RegionIndex': the index of the region inside the "regions" dictionary
    #               5- 'Adjacents': A list of adjacents. Two partitions X_1 and X_2 are said to be adjacent
    #                               if there exists x_1 \in X_1 and x_2 in X_2 such that
    #                               distance(x_1,x_2) < neighborhood_radius
    #
    ################################################################################################
    def __partition_statespace(self):
        if(self.num_integrators >1):
            for cube_index in range(len(self.cubes)):
                for region_index in range(len(self.regions)):
                    partitions = self.regions[region_index]['Partitions']
                    for partition_index in range(len(partitions)):
                        adjacents = []
                        for partition_adjacent in partitions[partition_index]['Adjacents']:
                            for cube_adjacent in self.cubes[cube_index]['Adjacents']:
                                adjacents.append(self.toSymbolicStateIndex(
                                    partition_adjacent['RegionIndex'],
                                    partition_adjacent['PartitionIndex'],
                                    cube_adjacent
                                ))

                        self.symbolic_states.append({
                            'Polygon': partitions[partition_index]['Polygon'],
                            'PolygonH': partitions[partition_index]['PolygonH'],
                            'HyperCube': self.cubes[cube_index]['HyperCube'],
                            'RegionIndex': region_index,
                            'PartitionIndex': partition_index,
                            'HyperCubeIndex': cube_index,
                            'SymbolicStateIndex' : len(self.symbolic_states),
                            'Adjacents': adjacents,
                            'isObstacle': partitions[partition_index]['isObstacle'],
                        })
                        if partitions[partition_index]['isObstacle'] == True:
                            self.obstacle_symbolic_states.append(len(self.symbolic_states)-1)
        
        else:
            for region_index in range(len(self.regions)):
                partitions = self.regions[region_index]['Partitions']
                for partition_index in range(len(partitions)):
                    adjacents = []
                    for partition_adjacent in partitions[partition_index]['Adjacents']:
                            adjacents.append(self.toSymbolicStateIndex(
                                partition_adjacent['RegionIndex'],
                                partition_adjacent['PartitionIndex'],
                                0
                            ))

                    self.symbolic_states.append({
                        'Polygon': partitions[partition_index]['Polygon'],
                        'PolygonH': partitions[partition_index]['PolygonH'],
                        # 'HyperCube': self.cubes[cube_index]['HyperCube'],
                        'RegionIndex': region_index,
                        'PartitionIndex': partition_index,
                        # 'HyperCubeIndex': 0,
                        'SymbolicStateIndex' : len(self.symbolic_states),
                        'Adjacents': adjacents,
                        'isObstacle': partitions[partition_index]['isObstacle'],
                    })

                    # if obstacle state, add its index to the obstacle_symbolic_states list
                    if partitions[partition_index]['isObstacle'] == True:
                        self.obstacle_symbolic_states.append(len(self.symbolic_states)-1)


        # # ----------------------------------------------------------------------------
        # #   STEP 1: Repartition the workspace regions based on the grid_size
        # # ----------------------------------------------------------------------------
        # for index in range(len(self.regions)):
        #     print(index, len(self.partitions))
        #     if self.regions[index]['isObstacle'] == False:
        #         # partition the workspace regions into cells
        #         region_partitions = self.__partition_region(index)
        #
        #         for region_partition in region_partitions:
        #             # Extend these cells in the higher-order state derivatives
        #             for cell_start_position in cell_positions:
        #                 hyper_cube_min = cell_start_position
        #                 hyper_cube_max = cell_start_position + self.grid_size
        #                 self.partitions.append({'Polygon': region_partition['Polygon'],
        #                                         'PolygonH': region_partition['PolygonH'],
        #                                         'HyperCube': {'min':hyper_cube_min, 'max':hyper_cube_max},
        #                                         'RegionIndex': region_partition['RegionIndex'],
        #                                         'HyperCubeIndex': [],
        #                                         'Adjacents': [],
        #                                         'isObstacle': region_partition['isObstacle']
        #                                         })
        #
        #     else: # Add the obstacles as cell partitions
        #         partition_polygon = self.regions[index]['Polygon']
        #         A,b = self.__computeHRepresentation(partition_polygon)
        #         for cell_start_position in cell_positions:
        #             hyper_cube_min = cell_start_position
        #             hyper_cube_max = cell_start_position + self.grid_size
        #             self.partitions.append({'Polygon': partition_polygon,
        #                                     'PolygonH': {'A': A, 'b': b},
        #                                     'HyperCube': {'min':hyper_cube_min, 'max':hyper_cube_max},
        #                                     'RegionIndex': index,
        #                                     'HyperCubeIndex': [],
        #                                     'Adjacents': [],
        #                                     'isObstacle': True
        #                                     })
        #
        #
        # print('checking for adjacents')
        # # ----------------------------------------------------------------------------
        # #   STEP 2: Check for Adjacent partitions
        # # ----------------------------------------------------------------------------
        # for index1 in range(len(self.partitions)):
        #     for index2 in range(index1+1, len(self.partitions)):
        #         min_distance, max_distance = self.__measureDistancePartitions(index1, index2)
        #
        #         if min_distance <= self.neighborhood_radius:
        #             self.partitions[index1]['Adjacents'].append(index2)
        #             self.partitions[index2]['Adjacents'].append(index1)
        #
        # if StateSpacePartioner_DEBUG:
        #     # Just for debugging: for each partition, plot the adjacent ones
        #     for index1 in range(len(self.partitions)):
        #         polygon = self.partitions[index1]['Polygon']
        #         fig = plt.figure(1, figsize=(5, 5), dpi=90)
        #         axis = fig.add_subplot(111)
        #         axis.set_title("Adjacents of partition " + str(index1))
        #         self.__plotWorkspacePolygon(polygon, str(index1), axis)
        #
        #         adjacents = self.partitions[index1]['Adjacents']
        #         for index2 in adjacents:
        #             polygon2 = self.partitions[index2]['Polygon']
        #             self.__plotWorkspacePolygon(polygon2, str(index2), axis)
        #
        #         plt.show()
        #





    ################################################################################################
    #
    #       Plot Workspace Regions, Partitions, and Polygons:
    #           Utility function that plots all the regions inside the "regions" data structure.
    #           It plots the index of the region inside the centroid of the region
    #
    ################################################################################################

    def __plotWorkspaceRegions(self):
        fig = plt.figure(1, figsize=(5,5), dpi=90)
        axis = fig.add_subplot(111)

        for index in range(len(self.regions)):
            polygon = self.regions[index]['Polygon']
            self.__plotWorkspacePolygon(polygon, str(index), axis)

        plt.show()



    def plotWorkspacePartitions(self, safe = [], pts=[], show  = False, traj = [], save_file = ''):
        fig = plt.figure(1, figsize=(5, 5), dpi=90)
        axis = fig.add_subplot(111)

        # for region_index in range(len(self.regions)):
        #     partitions = self.regions[region_index]['Partitions']
        #     for partition_index in range(len(partitions)):
        #         polygon = partitions[partition_index]['Polygon']
                # self.__plotWorkspacePolygon(polygon, str(region_index)+":"+str(partition_index), axis)
        for partition_idx, partition in enumerate(self.symbolic_states):
                polygon = partition['Polygon']
                self.__plotWorkspacePolygon(polygon, str(partition_idx), axis,color = 'r',alpha = 0.1)

        for partition_idx in safe:
            polygon = self.symbolic_states[partition_idx]['Polygon']
            color = 'g'
            fill = 'none'
            self.__plotWorkspacePolygon(polygon, str(partition_idx), axis, color,fill)
        
        for partition_idx in traj:
            polygon = self.symbolic_states[partition_idx]['Polygon']
            color = 'Y'
            fill = 'none'
            self.__plotWorkspacePolygon(polygon, str(partition_idx), axis, color,fill)

        if(len(pts)):
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            axis.scatter(x,y)     
        if(show):
            plt.show()
        else:
            plt.savefig(save_file)


    def __plotWorkspacePolygon(self, polygon, text, axis, color = 'r', fill = 'full', alpha = 1):
        x, y = polygon.exterior.xy
        axis.fill(x, y, color = color,edgecolor = 'black')

        center = polygon.centroid.coords[:][0]  # region center
        # axis.text(center[0], center[1], text,color =color)




    ################################################################################################
    #
    #       Compute Half Space Representation of a Polygon (A x <= b) using the Pycddlib package
    #
    ################################################################################################

    def __computeHRepresentation(self, polygon):
        # transform this intersection into half-space representation (A x <= b) using Pycddlib package
        x_partition_polygon, y_partition_polygon = polygon.exterior.xy
        x_partition_polygon.pop(0)  # remove the first element since it is equal to the last one
        y_partition_polygon.pop(0)  # remove the first element since it is equal to the last one


        # The Pycddlib asks to put a flag equal to "1" in the begining along with the x-y coordinates
        # of each point.
        points = [[1, x_partition_polygon[i], y_partition_polygon[i]] for i in range(len(x_partition_polygon))]


        matrix_cdd = cdd.Matrix(points, number_type='float')
        matrix_cdd.rep_type = cdd.RepType.GENERATOR
        poly_cdd = cdd.Polyhedron(matrix_cdd)

        polygon_halfspace = poly_cdd.get_inequalities()
        # this function returns the half-space representation one Matrix object with [b -A] elements inside

        b = np.empty([polygon_halfspace.row_size, 1])
        A = np.empty([polygon_halfspace.row_size, polygon_halfspace.col_size - 1])
        for row_index in range(polygon_halfspace.row_size):
            row = polygon_halfspace.__getitem__(row_index)
            b[row_index] = row[0]
            A[row_index, :] = [-1 * x for x in row[1:polygon_halfspace.col_size]]

        return A, b

    ################################################################################################
    #
    #       Given an hyper cube index, region index, and workspace partition index, return the state index
    #
    ################################################################################################

    #Think of a 2D matrix:
    #       Row#1 corresponds to all symbolic states that share cube 1,
    #       Row#2 corresponds to all symbolic states that share cube 2,
    #       ....
    #
    #       Column 1 corresponds to all symbolic states that share region partition 1:1
    #       Column 1 corresponds to all symbolic states that share region partition 1:2
    #       ....

    def toSymbolicStateIndex(self, regionIndex, partitionIndex, cubeIndex):
        row_index = cubeIndex
        column_index = self.__RegionPartitionIndexShift[regionIndex]+partitionIndex
        max_number_of_columns = self.__NumberOfRegionPartitions
        return column_index + row_index*max_number_of_columns


    # def __measureDistancePartitions(self, index1, index2):
    #     # compute the distance between the hypercube centers
    #     hypercube1_center = self.partitions[index1]['HyperCube']['min'] + 0.5 * self.grid_size
    #     hypercube2_center = self.partitions[index2]['HyperCube']['min'] + 0.5 * self.grid_size
    #
    #     distance_hypercube_centers = np.linalg.norm(np.subtract(hypercube1_center, hypercube2_center))
    #
    #     # distance between hypercube center and any vertex is equal to L*sqrt(n)/2 where L is the length of the
    #     # hypercube edges and n is the dimension of the hypercube (for n = 2 distance = 0.5L*sqrt(2))
    #     distance_center_vertex = self.grid_size * np.sqrt(len(hypercube1_center))/2
    #
    #     # compute the worst case (max and min) distances (happens when two vertices are on the
    #     # opposite sides of the center
    #     max_hypercube_distance = distance_hypercube_centers + 2 * distance_center_vertex
    #     min_hypercube_distance = np.amax([0.0, distance_hypercube_centers - 2 * distance_center_vertex])
    #
    #
    #     # add to it the distance between the workspace cells
    #     polygon1 = self.partitions[index1]['Polygon']
    #     polygon2 = self.partitions[index2]['Polygon']
    #
    #     min_distance = min_hypercube_distance + polygon1.distance(polygon2)
    #     max_distance = max_hypercube_distance + polygon1.distance(polygon2)
    #     return [min_distance, max_distance]  

