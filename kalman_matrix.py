import numpy as np
from numpy.linalg import inv

def kalman_m(x_k_i,F,B,U,Q,P,H,R,y):
    x_k = np.dot(F,x_k_i) + np.dot(B,U)
    P_k = np.dot(np.dot(F,P),inv(F))+Q
    
    K_k = np.dot(P_k,np.dot(H.T,inv(np.dot(np.dot(H,P),H.T)+R)))
    x_k_u = x_k + np.dot(K_k,(y-np.dot(H,x_k)))
    P_k_u = np.dot((np.eye(3)-np.dot(K_k,H)),P_k)
    
    return (x_k_u,P_k_u)
