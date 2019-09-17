from __future__ import print_function
import math
from constant import abs_tol


def is_close(a, b, rel_tol=1e-09, abs_tol=abs_tol):
    """
    Compare floats for equality
    This function is added in python 3.5 with default rel_tol=1e-09, abs_tol=0.0
    """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)



def is_between(aX, aY, bX, bY, cX, cY):
    """
    Check if point c (cX, cY) is strictly between points a (aX, aY) and b (bX, bY)
    """
    # a, b, c are aligned requires cross product of vectors (c-a) and (b-a) is zero
    cross_product = (cY - aY) * (bX - aX) - (cX - aX) * (bY - aY)
    if not is_close(cross_product, 0.0):
        return False

    # c between a, b requires dot product between vectors (c-a) and (b-a) is positive and 
    # less than than the square of distance beween a and b
    dot_product = (cX - aX) * (bX - aX) + (cY - aY) * (bY - aY)
    if dot_product + abs_tol < 0.0:
        return False

    squared_length_ba = (bX - aX) * (bX - aX) + (bY - aY) * (bY - aY)
    if dot_product > squared_length_ba + abs_tol:
        return False

    # c is not on a or b
    if is_close(cX, aX) and is_close(cY, aY) or is_close(cX, bX) and is_close(cY, bY):
        return False

    return True



def two_segments_intersection(s, t):
    """
    Find intersection between two segments s (endpoints a,b) and t (endpoints c,d)
    - s: (aX, aY, bX, bY)
    - t: (cX, cY, dX, dY)
    """
    aX, aY, bX, bY = s[0], s[1], s[2], s[3]
    cX, cY, dX, dY = t[0], t[1], t[2], t[3]

    A = (bX - aX) * (cY - dY) - (cX - dX) * (bY - aY)
    if not is_close(A, 0.0):
        As = (cX - aX) * (cY - dY) - (cX - dX) * (cY - aY)
        At = (bX - aX) * (cY - aY) - (cX - aX) * (bY - aY)
        ks = As / A
        kt = At / A
    else: 
        return False

    if 0.0-abs_tol <= ks <= 1.0+abs_tol and 0.0-abs_tol <= kt <= 1.0+abs_tol:
        intersection_X = (1 - ks) * aX + ks * bX
        intersection_Y = (1 - ks) * aY + ks * bY
    else: 
        return False

    return intersection_X, intersection_Y      


def print_region(reg_V):
    """
    Print all vertices of a V-rep polyhedron
    """
    for vertex in reg_V:
        print('{:04.2f}, {:04.2f}'.format(vertex[0], vertex[1]), end=' -> ')