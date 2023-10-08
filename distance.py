import numpy as np
def dist(a,b):
    dist = np.sum(np.abs(a-b)**2,axis=-1)**(1./2)
    return dist