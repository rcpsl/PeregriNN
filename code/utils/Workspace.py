from utility import *
from BinarySearchTree import *
import numpy as np
import json
import math

class Workspace(object):

    def __init__(self,num_vertices,num_refined_lasers, file):
        """
        Define workspace and LiDAR parameters
        """
        # LiDAR parameters
        self.max_range = 100.0

        # TODO: Currently does not support vertical (90 and 270) and horizontal (0 and 180) lasers due to using tan, cot.
        # NOTE: Laser angles are in range [0, 360) degrees.
        # NOTE: It might be good to define angles in precise. Otherwise, need to consider round-off error in sweep and minimal cycles

        ######################################
        # Use this configuration to generate regions AND experiments for Table 2-4
        ######################################
        num_vertices = 8
        self.principal_angles = []
        refined_angles = [82.87, 135, 262.87, 315,45, 97.13, 225, 277.13]
        
        # NOTE: Order of angles in laser_angles should be consistent with input image of NN 
        self.laser_angles = self.principal_angles + refined_angles
        self.laser_angles.sort()
        self.obstacles = []
        # print 'laser_angles = ', self.laser_angles
        # print 'Number of lasers = ', len(self.laser_angles)

        data = self.__parse_json(file)
        lines_json = data['Lines']
        vertices_json = data['vertices']
        lines = []
        for line_json in lines_json:
            orientation = 2
            if(line_json['orientation'] == 'V'):
                orientation = 1
            elif(line_json['orientation'] == 'H'):
                orientation = 0
            lines.append([line_json['x1'], line_json['y1'], line_json['x2'], line_json['y2'], orientation,line_json['index']])
        self.lines = lines

        vertices = []
        for vertex_json in vertices_json:
            obst = [self.lines[vertex_json['L1']], self.lines[vertex_json['L2']]]
            vertex = {'x': vertex_json['x'], 'y':vertex_json['y'], 'angle_lb': vertex_json['angle_lb'], 
                        'angle_ub': vertex_json['angle_ub'], 'obst': obst}
            vertices.append(vertex)
        self.vertices = vertices

        #Parse obstacles
        obstalces_json = data['obstacles']
        for obst_json in obstalces_json:
            vertices = obst_json["vertices"]
            self.obstacles.append(vertices)
                


    def prepare_workspace(self):    
        """
        Prepare workspace for sweep algorithm 
        """
        events, segments = [], []
        event_queue  = EventQueue()

        # Create events and segments correspond to obstacle boundaries
        for obst in self.lines:
            if obst[4] == 1: # vertical 
                angle = 90.0
                # Upper endpoint has bigger y-coordinate
                upper_X, upper_Y, lower_X, lower_Y = obst[2], obst[3], obst[0], obst[1]
            elif obst[4] == 0:
                angle = 180.0
                # Upper endpoint has smaller x-coordinate
                upper_X, upper_Y, lower_X, lower_Y = obst[0], obst[1], obst[2], obst[3]
            else:
                # Upper endpoint has smaller x-coordinate
                upper_X, upper_Y, lower_X, lower_Y = obst[2], obst[3], obst[0], obst[1]
                dy = upper_Y - lower_Y
                dx = upper_X - lower_X
                angle = math.atan2(dy,dx) * 180 / math.pi

            new_segment = Segment(upper_X, upper_Y, lower_X, lower_Y, angle, is_principal=True, is_boundary=True)
            segments.append(new_segment)

            # Create event for upper endpoint
            upper_event = Event(upper_X, upper_Y, set(), set(), set(), is_principal=True)
            new_upper_event = event_queue.search(event_queue.root, upper_event)
            if new_upper_event is None:
                new_upper_event = upper_event
                event_queue.insert(new_upper_event)
                events.append(new_upper_event)
            new_upper_event.U.add(new_segment)

            # Create event for lower endpoint
            lower_event = Event(lower_X, lower_Y, set(), set(), set(), is_principal=True)
            new_lower_event = event_queue.search(event_queue.root, lower_event)
            if new_lower_event is None:
                new_lower_event = lower_event
                event_queue.insert(new_lower_event)
                events.append(new_lower_event)
            new_lower_event.L.add(new_segment)


        # Partition workspace by segments start from vertices, along laser directions, and in the angle range of corresponding vertices
        for vertex_index, vertex in enumerate(self.vertices):
            if vertex['angle_lb'] <= vertex['angle_ub']:
                vertex_laser_angles = [x for x in self.laser_angles if vertex['angle_lb'] <= x <= vertex['angle_ub']]
            else:
                vertex_laser_angles = [x for x in self.laser_angles if 0.0 <= x <= vertex['angle_ub']]
                vertex_laser_angles.extend([x for x in self.laser_angles if vertex['angle_lb'] <= x < 360.0])

            # Initiate endpoints of partition segments by laser max range
            # TODO: Use radians for angles instead of conversion angles repeatedly for all vertices
            laser_ends_X = [vertex['x'] + self.max_range * math.cos(math.radians(angle)) for angle in vertex_laser_angles]
            laser_ends_Y = [vertex['y'] + self.max_range * math.sin(math.radians(angle)) for angle in vertex_laser_angles]

            # Compute intersection between obstacles and partition segments start from this vertex.
            # TODO: Computation could be more efficient by considering obstacles are either vertical or horizontal.
            # TODO: Add a check to prevent partition segments overlap obstacle boundaries.
            for i in xrange(len(vertex_laser_angles)):
                for obst in self.lines:
                    if obst not in vertex['obst']: 
                        obst_start_X, obst_start_Y, obst_end_X, obst_end_Y = obst[0], obst[1], obst[2], obst[3]
                        end_X, end_Y = laser_ends_X[i], laser_ends_Y[i]            
                        intersection = two_segments_intersection((obst_start_X, obst_start_Y, obst_end_X, obst_end_Y), (vertex['x'], vertex['y'], end_X, end_Y))

                        if intersection:
                            intersection_X, intersection_Y = intersection[0], intersection[1]
                            laser_ends_X[i] = intersection_X
                            laser_ends_Y[i] = intersection_Y 


            # Search event correspond to this vertex
            vertex_event     = Event(vertex['x'], vertex['y'], set(), set(), set(), is_principal=True)
            new_vertex_event = event_queue.search(event_queue.root, vertex_event)
            assert new_vertex_event is not None, 'all vertex events should have been enqueued when process obstacle boundaries'
            #if new_vertex_event is None:
            #    new_vertex_event = vertex_event
            #    event_queue.insert(new_vertex_event)
            #    events.append(new_vertex_event)

            for i, angle in enumerate(vertex_laser_angles):
                is_principal = True if angle in self.principal_angles else False
                # Segement with both endpoints are vertices should only be count once in order to avoid overlapping segments
                endpoint_is_processed_vertex = False
                for j in xrange(vertex_index):
                    if is_close(laser_ends_X[i], self.vertices[j]['x']) and is_close(laser_ends_Y[i], self.vertices[j]['y']):
                        endpoint_is_processed_vertex = True
                        break

                if not endpoint_is_processed_vertex:
                    # Create an event for partition segment endpoint.
                    # NOTE: is_principal of endpoint could true for a non-principle laser if the endpoint is a vertex. 
                    # However, all vertices should have ALREADY been enqueued
                    endpoint_event     = Event(laser_ends_X[i], laser_ends_Y[i], set(), set(), set(), is_principal=is_principal)
                    new_endpoint_event = event_queue.search(event_queue.root, endpoint_event)
                    if new_endpoint_event is None:
                        new_endpoint_event = endpoint_event
                        event_queue.insert(new_endpoint_event)
                        events.append(new_endpoint_event)
                        
                    # Create a segment and update events correspond to its endpoints     
                    if 0.0 < angle <= 180.0: # vertex is lower endpoint of the new segment
                        new_segment = Segment(vertex['x'], vertex['y'], laser_ends_X[i], laser_ends_Y[i], angle, is_principal=is_principal, is_boundary=False)
                        new_vertex_event.L.add(new_segment)
                        new_endpoint_event.U.add(new_segment)
                    else: # vertex is upper endpoint of the new segment
                        aligned_angle = (angle + 180.0) % 360
                        new_segment = Segment(vertex['x'], vertex['y'], laser_ends_X[i], laser_ends_Y[i], aligned_angle, is_principal=is_principal, is_boundary=False)
                        new_vertex_event.U.add(new_segment)
                        new_endpoint_event.L.add(new_segment)
                    segments.append(new_segment)

        return events, segments, event_queue


    def __parse_json(self,file):
        with open(file) as json_file:  
            data = json.load(json_file)
            return data
    def find_lidar_configuration(self, subdivisions):
        """
        LiDAR configuration for each subdivision is represented by a list of obstacle indices that lasers intersect.
        NOTE: The order of intersecting obstacle indices should be same as that of lasers in laser_angles.
        - subdivisions: a list of polyhedra, in V-rep
        """    
        lidar_configs = []
        # NOTE: lidar_configs has the same order as regions in subdivisions
        for poly in subdivisions:
            # Take an interior point of the subdivision and compute intersecting obstacle indices
            interior_X = (poly[0][0] + poly[1][0] + poly[2][0]) / 3.0
            interior_Y = (poly[0][1] + poly[1][1] + poly[2][1]) / 3.0

            # TODO: Convert angle to radians out of loop
            laser_ends_X = [interior_X + self.max_range * math.cos(math.radians(angle)) for angle in self.laser_angles]
            laser_ends_Y = [interior_Y + self.max_range * math.sin(math.radians(angle)) for angle in self.laser_angles]

            configuration = []
            for i in xrange(len(self.laser_angles)):
                closest_obst_index = -1
                for obst in self.lines:
                    obst_start_X, obst_start_Y, obst_end_X, obst_end_Y = obst[0], obst[1], obst[2], obst[3]
                    end_X, end_Y = laser_ends_X[i], laser_ends_Y[i]            
                    intersection = two_segments_intersection((obst_start_X, obst_start_Y, obst_end_X, obst_end_Y), (interior_X, interior_Y, end_X, end_Y))

                    if intersection:
                        intersection_X, intersection_Y = intersection[0], intersection[1]
                        laser_ends_X[i] = intersection_X
                        laser_ends_Y[i] = intersection_Y 
                        closest_obst_index = obst[5]

                assert closest_obst_index != -1, 'No intersecting obstacle is found when compute LiDAR configuration'
                configuration.append(closest_obst_index)          

            lidar_configs.append(configuration)    

        return lidar_configs

