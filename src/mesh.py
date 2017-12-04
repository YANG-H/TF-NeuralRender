import pymesh as pm
import numpy as np

def load_obj(fname):
    with open(fname, 'r') as f:
        vdata = []
        vtdata = []
        fdata = []
        fvtdata = []
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue
            if tokens[0] == '#':
                continue
            if tokens[0] == 'v':
                vdata += [[float(x) for x in tokens[1:]]]
                continue
            if tokens[0] == 'vt':
                vtdata += [[float(x) for x in tokens[1:]]]
                continue
            if tokens[0] == 'f':
                for idx in tokens[1:]:
                    ids = map(x:int(x), filter(x:x != '', idx.split('/')))
                    
                    
        

