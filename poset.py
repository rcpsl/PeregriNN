import cdd
import numpy as np
from time import time
import scipy
import scipy.optimize
import sys
from copy import copy
from queue import Queue
import operator as op
from functools import reduce



def minimal_H(A, b, bounds):
    #check feasibility first
    try:
        res = scipy.optimize.linprog(np.ones(A.shape[1]),A,b,bounds=bounds,method='interior-point')
        if(res.success == False):
            return None, None,None

        to_keep = []
        for c_idx in range(len(A)):
            A[c_idx] = -A[c_idx]
            b[c_idx] = -b[c_idx]
            res = scipy.optimize.linprog(np.zeros(A.shape[1]),A,b,bounds=bounds,method='interior-point')
            if(res.success == True):
                to_keep.append(c_idx)
            A[c_idx] = -A[c_idx]
            b[c_idx] = -b[c_idx]
        
        A = A[to_keep]
        b = b[to_keep]
    except Exception as e:
        print(len(A), len(b),e)
    return (A,b, to_keep)

def compute_successors(root):

    # if(len(root.children) == 0):
    #     root.num_successors = 0
    #     return 0
    # successors = 1
    # for child in root.children:
    #     successors += compute_successors(child)
    # root.num_successors = successors
    # return successors
    if(len(root.children) == 0):
        root.num_successors = 0
        return set()
    successors = set()
    for child in root.children:
        successors.add(child.hash_val)
        successors = successors.union(compute_successors(child))
    root.num_successors = len(successors)
    return successors

class PosetNode(object):
    def __init__(self, A, b, input_domain, bounding_faces_idx, fixed_faces = set()):
        self.children = []
        self.region = {'A':A, 'b': b}
        self.fixed_faces = fixed_faces
        self.num_successors = 0
        self.input_domain = input_domain
        self.faces = bounding_faces_idx
        self.hash_val = self.hash()
        self.visited = False

    def hash(self):
       return hash(frozenset(self.fixed_faces))

    

class Poset(object):
    def __init__(self, input_domain, A, b,pt):
        #input bounding box
        self.input_domain = input_domain
        self.A = A
        self.b = b 
        self.pt = pt.reshape((-1,1))
        self.hashMap = {}
        A_,b_, minimal_set = self.minimal_H_cdd(self.A, self.b, self.input_domain)
        # if(len(minimal_set) == 0):
        #     print(A)
        #     print(b)
        #     print(input_domain)
        self.root = PosetNode(A_, b_, self.input_domain, minimal_set)


    def build_poset(self):
        if(self.root == None):
            return

        queue = Queue()
        hashMap = {}
        queue.put(self.root)
        hashMap[self.root.hash_val] = self.root
        while(not queue.empty()):
            pNode = queue.get()
            #Generate all children
            #Add them to the Queue
            children = self.get_neighbours(pNode)
            for idx, child in enumerate(children):
                if child.hash_val in hashMap:
                    children[idx] = hashMap[child.hash_val]
                else:
                    hashMap[child.hash_val] = child
                    queue.put(child)
            pNode.children = children
            # children = [child for child in children if child.hash_val not in hashMap]
            # map(queue.put, children)
        self.hashMap = hashMap


    
    def get_neighbours(self, pNode):
        neighbours = []
        A_all, b_all = copy(self.A), copy(self.b)

        for f_idx in pNode.fixed_faces:
            A_all[f_idx] *= -1
            b_all[f_idx] *= -1

        for f_idx in range(len(pNode.faces)):
            face_idx = pNode.faces[f_idx]
            A_all[face_idx] = pNode.region['A'][f_idx]
            b_all[face_idx] = pNode.region['b'][f_idx]
    
        for f_idx in range(len(pNode.faces)):
            face_idx = pNode.faces[f_idx]
            if(face_idx in pNode.fixed_faces):
                continue
            #flip one face
            A_all[face_idx] = -pNode.region['A'][f_idx]
            b_all[face_idx] = -pNode.region['b'][f_idx]

            A_,b_, minimal_set = self.minimal_H_cdd(A_all,b_all,self.input_domain)
            if(A_ is None):
                continue
            fixed_faces  = copy(pNode.fixed_faces)
            fixed_faces.add(face_idx)
            neighbours.append(PosetNode(A_, b_, self.input_domain, minimal_set, fixed_faces))
            #flip back
            A_all[face_idx] = -A_all[face_idx]
            b_all[face_idx] = -b_all[face_idx]

        return neighbours

    def minimal_H_cdd(self,A,b,bounds):
        if(len(self.hashMap) == 0):
            if(len(bounds) > 0):
                A_ = np.vstack((np.eye(A.shape[1]),-np.eye(A.shape[1])))
                b_ = np.vstack((bounds[:,1].reshape((-1,1)),-bounds[:,0].reshape((-1,1))))
                idxs = np.where(A_.dot(self.pt) > b_)[0]
                A_[idxs] = -A_[idxs]
                b_[idxs] = -b_[idxs]
                A_ = np.vstack((A,A_))
                b_ = np.vstack((b,b_))
            else:
                A_ = A
                b_ = b
        else:
            if(len(bounds) >0 ):
                A_ = np.vstack((A,np.eye(A.shape[1])))
                A_ = np.vstack((A_,-np.eye(A.shape[1])))
                b_ = np.vstack((b,bounds[:,1].reshape((-1,1))))
                b_ = np.vstack((b_,-bounds[:,0].reshape((-1,1))))
            else:
                A_ = A
                b_ = b
        H = np.hstack((b_,-A_))
        mat = cdd.Matrix(H)
        mat.rep_type = cdd.RepType.INEQUALITY
        ret = mat.canonicalize()
        to_keep = sorted(list(frozenset(range(len(A))) - ret[1]))
        to_keep_region = sorted(list(frozenset(range(len(A_))) - ret[1]))
        # return A[to_keep], b[to_keep], to_keep
        return A_[to_keep_region], b_[to_keep_region], to_keep
        pass



def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

if __name__ == "__main__":
    input_dim = 5
    input_domain = np.array([[-1,1]] * input_dim)
    input_domain = []
    W = np.array([[1,-1],[-1,-1],[1,-1]])
    b = np.array([-1,0,0]).reshape((-1,1))
    for dim in [10]:
        W = np.random.normal(0,3,size = (dim,input_dim))
        b = np.random.normal(0,3,size = (dim,1)) 
        c = W.dot(np.zeros((input_dim,1))) + b
        idxs = np.where(c < 0)[0]
        W[idxs] = -W[idxs]
        b[idxs] = -b[idxs]

        s_t = time()
        poset = Poset(input_domain,W,b,np.zeros((input_dim,1)))
        e_t = time()
        # print('minimal H rep time:', e_t-s_t)
        poset.build_poset()
        # compute_successors(poset.root)
        f_t = time()
        exact_nodes = sum([ncr(dim,d) for d in range(input_dim+1)])
        print('Dim:',dim,'nodes:',len(poset.hashMap), 'Minimal H rep:',e_t-s_t,'Total time:', f_t-s_t)
        pass
        for k,node in poset.hashMap.items():
            print(set(range(W.shape[0])) - node.fixed_faces, node.num_successors)
        queue = Queue()
        queue.put(poset.root)
        while(not queue.empty()):
            node = queue.get()
            map(queue.put,node.children)
            print(set([0,1,2]) - node.fixed_faces,len(node.children),node)
        pass


    


