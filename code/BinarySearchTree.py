from __future__ import print_function
#from functools import total_ordering
import sys
from utility import *

class Node(object):
    def __init__(self):
        self.parent = None
        self.left   = None
        self.right  = None



# NOTE: Instead of using @total_orderting, explicitely define comparison methods to support precision tolerence.
# With @total_ordering, need to define __eq__, __ne__, __lt__
#@total_ordering
class Event(Node):
    """
    Node for event queue
    """
    def __init__(self, x, y, U, L, C, is_principal):
        """
        - x, y    : X, Y coordinates of event point
        - U, L, C : sets of segments whose upper, lower endpoint, or interior point is this event, respectively
        - is_principal: true if at least two principal partition segments (obstacle boundaries or principal lasers) intersect at this point
        """
        super(Event, self).__init__()
        self.x = x
        self.y = y
        self.U = U
        self.L = L
        self.C = C
        self.is_principal = is_principal
        # A set of indices of abstract regions that the event is contained (include on boundary)
        self.abst_reg = set()
    
    def __str__(self):
        #return str(self.x) + ', ' + str(self.y)
        return str('{:04.2f}'.format(self.x)) + ', ' + str('{:04.2f}'.format(self.y))

    # Event points p lt q if and only if p_y > q_y holds or p_y = q_y and p_x < q_x
    def __eq__(self, other):
        return is_close(self.y, other.y) and is_close(self.x, other.x)

    def __ne__(self, other):
        return not self.__eq__(other)
            
    def __lt__(self, other):
        return self.y > other.y + abs_tol or is_close(self.y, other.y) and self.x + abs_tol < other.x

    def __gt__(self, other):
        sys.exit('Exit because Event __gt__ is not implemented')

    def __le__(self, other):
        sys.exit('Exit because Event __le__ is not implemented')

    def __ge__(self, other):
        sys.exit('Exit because Event __ge__ is not implemented')      



#@total_ordering
class Segment(Node):
    """
    Node for sweep line status structure
    """
    def __init__(self, x1, y1, x2, y2, angle, is_principal, name='s', is_boundary=False):
        """
        - segment endpoints (x1, y1), (x2, y2), order of the two points does not matter
        - angle beween line and X positive, should be in range (0, 180]
        - is_principal: true if this segment corresponds to a principal laser or obstacle boundary    
        - name: only used for printing segment when debug
        - is_boundary: boolean, True if the edge corresponds to workspace or obstacle boundary.
                      Only used in searching minimal cycles
        """    
        super(Segment, self).__init__()
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.angle = angle
        self.is_principal = is_principal
        self.name = name
        self.is_boundary = is_boundary
        # upper, lower, interior: upper endpoint, lower endpoint, and a list of interor events on a segment, respectively.
        # They are are only used in searching minimal cycles, but not in sweep algorithm  
        self.upper  = None
        self.lower  = None
        self.interior = []

        # NOTE: angle is given and hence precise enough for equality comparison
        if self.angle == 180.0 or self.angle == 90.0:
            self.slope     = None
            self.intercept = None
        else:
            self.slope     = (y2 - y1) / (x2 - x1)
            self.intercept = y1 - self.slope * x1
        # x-oordinate of intersection point between this segment and sweep line
        self.sweep_intersection_X = None

    def __str__(self):
        return str('{:04.2f}'.format(self.x1)) + ', ' + str('{:04.2f}'.format(self.y1)) + ' -- ' + str('{:04.2f}'.format(self.x2)) + ', ' + str('{:04.2f}'.format(self.y2))


    def evaluate_sweep_intersection(self, event):
        if self.angle == 180.0:
            self.sweep_intersection_X = event.x
        elif self.angle == 90.0:
            self.sweep_intersection_X = self.x1
        else: 
            self.sweep_intersection_X = (event.y - self.intercept) / self.slope


    # TODO: The CG book does not define an ordering for segments that intersect sweep line at the same point.
    # TODO: Comparison between x-coordinates of intersection between sweep line and segments can be done by cross product.
    # Segments s1 lt s2 if and only if x-coordinate of intersection between s1 and sweep line is less than
    # that of s2, or angle of s1 is less than that of s2 when they intersect sweep line at the same point
    def __eq__(self, other):
        # NOTE: Even have assumed no partially overlapping segments, it is insufficient to decide equality between
        # segments based on intersection point with sweep line and angle. For example, when a point is 
        # lower endpoint of one segment and upper endpoint of another, and these two segments have same angle.
        # However, this equality comparison is sufficient in this sweep algrithm since no comparison between
        # a segment ends at a point and a segment starts from the same point will ever happen
        return is_close(self.sweep_intersection_X, other.sweep_intersection_X) and self.angle == other.angle

    def __ne__(self, other):
        return not self.__eq__(other)      

    def __lt__(self, other):
        return self.sweep_intersection_X + abs_tol < other.sweep_intersection_X or \
                is_close(self.sweep_intersection_X, other.sweep_intersection_X) and self.angle < other.angle 

    def __gt__(self, other):
        sys.exit('Exit because Segment __gt__ is not implemented')

    def __le__(self, other):
        sys.exit('Exit because Segment __le__ is not implemented')

    def __ge__(self, other):
        sys.exit('Exit because Segment __ge__ is not implemented')           



class BinarySearchTree(object):
    def __init__(self):
        self.root = None
        self.BSTlist = []

    def search(self, x, y):
        while x is not None and y != x:
            if y < x:
                x = x.left
            else:
                x = x.right
        return x

    def minimum(self, x):
        while x.left is not None:
            x = x.left
        return x    

    def maximum(self, x):
        while x.right is not None:
            x = x.right
        return x     

    def successor(self, x):
        if x.right is not None:     
            return self.minimum(x.right)
        y = x.parent
        while y is not None and x is y.right:
            x = y
            y = y.parent
        return y     

    def predecessor(self, x):     
        if x.left is not None:
            return self.maximum(x.left)
        y = x.parent
        while y is not None and x is y.left:
            x = y
            y = y.parent
        return y    

    def insert(self, z):
        y = None
        x = self.root
        while x is not None:
            y = x
            if z < x:
                x = x.left
            else:
                x = x.right
        z.parent = y
        if y is None:
            self.root = z
        elif z < y:
            y.left = z
        else:
            y.right = z         

    def delete(self, z):
        if z.left is None:
            self.transplant(z, z.right)
        elif z.right is None:
            self.transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            if y.parent is not z:
                self.transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self.transplant(z, y)
            y.left = z.left
            y.left.parent = y      

    def transplant(self, u, v):
        if u.parent is None:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v is not None:
            v.parent = u.parent                



class EventQueue(BinarySearchTree):
    def __init__(self):
        super(EventQueue, self).__init__()

    def inorder_walk(self, node):
        if node is not None:
            self.inorder_walk(node.left)
            print((node.x, node.y))
            self.inorder_walk(node.right)



class SweepStatus(BinarySearchTree):
    def __init__(self):
        super(SweepStatus, self).__init__()

    def update_sweep_intersections(self, node, event):
        # Update X-coordinates at which sweep line intersects segments stored in sweep status
        if node is not None:
            self.update_sweep_intersections(node.left, event)
            node.evaluate_sweep_intersection(event)
            self.update_sweep_intersections(node.right, event)
            
    def inorder_walk(self, node):
        if node is not None:
            self.inorder_walk(node.left)
            print(node.name, end=', ')
            self.inorder_walk(node.right)
