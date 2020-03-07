from copy import copy
from time import time

SAT = True
UNSAT = False

class QuickXplain(object):
    def __init__(self,predicate):
        self.check = predicate
    

    #Inputs:
    #C :set of constraints initially empty
    #U :set of conflicting constraints
    #B : Background constraints which have to be satisfied
    def quickxplain(self,C,U):
        if(len(C) !=0 and self.check(C) == UNSAT):
            return []
        if(len(U) == 0):
            return []
        C_temp = copy(C)
        for idx, cstrnt in enumerate(U):
            C_temp += [cstrnt]
            sat = self.check(C_temp)
            if(not sat):
                break
        if(sat):
            return []
        
        X = [cstrnt]
        split_idx = idx/2
        U1 = U[:split_idx+1]
        U2 = U[split_idx+1:idx]
        if(len(U2) != 0):
            C_temp = C + U1 + X
            X2 = self.quickxplain(copy(C_temp), copy(U2))
            X = X + X2
        if(len(U1) != 0):
            C_temp = C + X
            X1 = self.quickxplain(copy(C_temp), copy(U1))
            X = X + X1
    
        return X



        
        


def check(C):
    return bool(int(raw_input('')))

# q = QuickXplain(check)
# U = set([('y_0',1),('y_1',1),('y_3',0),('y_4',0)])
# q.quickxplain(set(),U)

