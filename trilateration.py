import numpy as np

def trilat(a1,a2,a3,dist,dim):
    r = len(dist)
    tag_est = np.empty([r,dim])
    A = 2*np.array([a2-a1,
                   a3-a2,
                   a1-a3])
    
    for i in range(r):
        dist_mat = np.array([[dist[i][0]**2-dist[i][1]**2+a2[0]**2-a1[0]**2+a2[1]**2-a1[1]**2],
                         dist[i][1]**2-dist[i][2]**2+[a3[0]**2-a2[0]**2+a3[1]**2-a2[1]**2],
                         dist[i][2]**2-dist[i][0]**2+[a1[0]**2-a3[0]**2+a1[1]**2-a3[1]**2]])
        tag_est_i = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T),dist_mat)
        tag_est[i,:] = tag_est_i.T
    
    return tag_est