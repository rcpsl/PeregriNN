import cdd
import numpy as np

def get_bb(region):
    A,b = region['A'], region['b']
    H_rep = np.hstack((b,-A))
    m = cdd.Matrix(H_rep)
    m.rep_type = 1
    p = cdd.Polyhedron(m)
    new_bound = None

    try:
        vertices = np.array(p.get_generators())[:,1:]
        hrect_min = np.min(vertices,axis = 0).reshape((-1,1))
        hrect_max = np.max(vertices,axis = 0).reshape((-1,1))
        new_bound = np.hstack((hrect_min,hrect_max))
    except Exception as e:
        pass
    return new_bound