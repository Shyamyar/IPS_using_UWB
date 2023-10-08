#%%
import serial
import numpy as np
# import pandas as pd
import math
import matplotlib.pyplot as plt
import trilateration as tl
import kalman_matrix as kfm
#from conf_ellipse import confidence_ellipse
from distance import dist

uwb = serial.Serial("/dev/ttyACM0",115200,timeout=None)
# uwb = open("test_data","r")
# uwb_file = open("test_data2","w")

#%% Sensor Numbers and Positions
c = 3 # number of sensors (columns)
r = 200 # time/number of data (rows)
dim = 2
a1 = np.array([0,0]) #D532, 1082
a2 = np.array([3*0.46,-0.46]) #5D00, 17FF
a3 = np.array([4*0.46,2*0.46]) #9C23, 20B5
# a1 = np.array([0,-0.21]) #D532, 1082
# a2 = np.array([-0.2,1.27]) #5D00, 17FF
# a3 = np.array([1.70,1.2]) #9C23, 20B5
tag_orig = np.array([2*0.46,0.46])
D_true_i = np.array([dist(a1,tag_orig),dist(a2,tag_orig),dist(a3,tag_orig)])
D_true = np.tile(D_true_i,(r,1)) # Actual distance between sensors and tag

#%% Sensor ID Generator
ids = [None]*c
count = 0
while True:
    count+=1
    line=uwb.readline().decode()
    print(line)
    equal_pos = []
    space_pos = []
    if len(line)>=(c*25)+2:
        for i in range(len(line)):
            if (line[i] == "="):
                equal_pos.append(i)
        for i in range(len(line)):
            if (line[i] == " "):
                space_pos.append(i)
        for k in range(c):
            if k == 0:
                ids[k]=str(line[0:4])
            else:
                ids[k]=str(line[space_pos[k-1]+1:space_pos[k-1]+5])
        break
    else:
        continue

# print(ids)

#%% Filtering
D_M = np.empty((r,c)) # Distances measured from sensors over time
D_M[:] = np.NaN
D_est = np.empty([r,c]) # Distances estimated from kalman for sensors over time
D_est[:] = np.NaN
P_D = np.empty([r,c]) # Covariances (intra-state) estimated from kalman for sensors over time
P_D[:] = np.NaN

F = np.eye(3) # State Transition Matrix
B = np.zeros([3,3]) # Control Transition Matrix
U = np.zeros(3) # Control
Q = np.zeros([3,3]) # Noise
P = abs(np.random.normal(0,0.01)*np.eye(3)) # Estimation Error Matrix
H = np.eye(3) # Observation Matrix
R = 0.1*np.eye(3) # Measurement Error Matrix

i=0
while i<r:
    
    data = uwb.readline().decode()
    # uwb_file.write(data)
    print(data)
    for j in range(c):
        try:
            id_check = str(data[equal_pos[j]-20:equal_pos[j]-16])
            pos = ids.index(id_check)
            D_M[i][pos] = float(data[equal_pos[j]+1:space_pos[j]])
        except:
            continue
    
    if i==0:
        D_est[i,:] = D_M[i,:]
        P_D[i,:] = [P[0,0],P[1,1],P[2,2]]
        x_k_i = D_M[i,:]
        i+=1
        continue
    
    y = D_M[i,:]
    
    for j in range(c):
        if math.isnan(y[j]):
            y[j] = D_M[i-1,j]
            continue
    x_k, P_est = kfm.kalman_m(x_k_i,F,B,U,Q,P,H,R,y)
    P = P_est
    x_k_i = x_k
    D_est[i,:] = x_k
    P_D[i,:] = [P_est[0,0],P_est[1,1],P_est[2,2]]
    i+=1

# print(D_M)
# Dis_M = pd.DataFrame(D_M,columns=ids)
# print(Dis_M)
# average_dist = np.nanmean(Dis_M.iloc[:,range(c)],axis=0)

uwb.close()
# uwb_file.close()

#%% Without Filter
tag_est1 = tl.trilat(a1,a2,a3,D_M,dim)
# print(tag_est1)
Error1 = dist(tag_est1,tag_orig)

#%% With Filter        
tag_est2 = tl.trilat(a1,a2,a3,D_est,dim)
# print(tag_est2)
Error2 = dist(tag_est2,tag_orig)

#%% Confidence Interval (3 sigma)
D_M_error = D_M-D_true
D_est_error = D_est-D_true
ci_pos = D_est_error+3*np.sqrt(P_D)
ci_neg = D_est_error-3*np.sqrt(P_D)

#%% Plots
start = 5

plt.scatter(tag_est1[start:,0],tag_est1[start:,1],label='Unfiltered')
plt.scatter(tag_est2[start:,0],tag_est2[start:,1],label='Filtered')
plt.scatter(tag_orig[0],tag_orig[1],label='True Position')
ax = plt.gca()
confidence_ellipse(tag_est1[start:,0],tag_est1[start:,1], ax, n_std=1, edgecolor='C0', linestyle=':', label='$1\sigma$')
confidence_ellipse(tag_est1[start:,0],tag_est1[start:,1], ax, n_std=2, edgecolor='C1', linestyle=':', label='$2\sigma$')
confidence_ellipse(tag_est1[start:,0],tag_est1[start:,1], ax, n_std=3, edgecolor='C2', linestyle=':', label='$3\sigma$')
confidence_ellipse(tag_est2[start:,0],tag_est2[start:,1], ax, edgecolor='orange', linestyle=':')
plt.title('Positioning for Tag')
plt.xlabel('Time (sec)')
plt.ylabel('Distance (m)')
plt.legend()
plt.show()

plt.plot(np.array(range(start,r))/10,Error1[start:],label='Unfiltered')
plt.plot(np.array(range(start,r))/10,Error2[start:],label='Filtered')
plt.title('Positioning Error w.r.t True Tag Position')
plt.xlabel('Time (sec)')
plt.ylabel('Distance (m)')
plt.legend()
plt.show()

sensor = 2
plt.plot(np.array(range(start,r))/10,D_M[start:,sensor],label='Unfiltered')
plt.plot(np.array(range(start,r))/10,D_est[start:,sensor],label='Filtered')
plt.title('Distance from sensor '+str(ids[sensor]))
plt.xlabel('Time (sec)')
plt.ylabel('Distance (m)')
plt.legend()
plt.show()

plt.plot(np.array(range(start,r))/10,D_M_error[start:,sensor],label='Error Unfiltered')
plt.plot(np.array(range(start,r))/10,D_est_error[start:,sensor],label='Error Filtered')
plt.plot(np.array(range(start,r))/10,ci_pos[start:,sensor],label='$3\sigma$')
plt.plot(np.array(range(start,r))/10,ci_neg[start:,sensor],label='$-3\sigma$')
plt.title('Distance Error for sensor '+str(ids[sensor]))
plt.xlabel('Time (sec)')
plt.ylabel('Distance (m)')
plt.legend()
