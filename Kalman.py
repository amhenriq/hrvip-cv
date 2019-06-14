from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pykalman import KalmanFilter

df=pd.read_csv('datafile.csv', header=None, delim_whitespace=True)
df = DataFrame(df)#.fillna(0)
df.columns = ['Time', 'X', 'Y', 'Z'] #renames 0,1,2 to time, xcord, ycord

file = open("kalman_data.csv", "w")

time = df.Time
dt_init = time[1]-time[0]
dt1 = time[2]-time[1]
dt2 = time[3]-time[2]
print(dt_init)
print(dt1)
print(dt2)
x = df.X
y = df.Y

##############################
plt.title('X and Y Positions')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x,y)
#plt.legend() 
plt.show()
##############################
###concatenate x and y columns###
measured = pd.concat([x, y], axis=1)
###mask nan values###
measured_mask = np.ma.masked_invalid(measured) 

###KALMAN STUFF###
Transition_Matrix=[[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]  # A matrix
Observation_Matrix=[[1,0,0,0],[0,1,0,0]] # H matrix

xinit=measured_mask[0,0] ## First Measurement of x-coord
yinit=measured_mask[0,1] ## First Measurement of y-coord
vxinit=(measured_mask[1,0]-measured_mask[0,0]) ## as v = (d_1 - d_0)/(time)
vyinit=(measured_mask[1,1]-measured_mask[0,1]) 
initstate=[xinit,yinit,vxinit,vyinit]
initcovariance=1.0e-3*np.eye(4) 
transistionCov=1.0e-4*np.eye(4)
observationCov=1.0e-1*np.eye(2)
kf=KalmanFilter(transition_matrices=Transition_Matrix,
            observation_matrices =Observation_Matrix,
            initial_state_mean=initstate,
            initial_state_covariance=initcovariance,
            transition_covariance=transistionCov,
            observation_covariance=observationCov)

(filtered_state_means, filtered_state_covariances) = kf.filter(measured_mask)

x1 = pd.DataFrame(filtered_state_means[:,0])
y1 = pd.DataFrame(filtered_state_means[:,1])
output = pd.concat([x1, y1], axis=1)

csv = output.to_csv('kalman_data.csv', sep='\t')

#file = file.assign(X=pd.Series(x1))
#file = file.assign(Y=pd.Series(y1))
#output = output.applymap(str)
#file.write(output)


plt.plot(measured_mask[:,0],measured_mask[:,1],'r',label='measured')
#plt.axis([-.05,.35,-.1,.15])
plt.plot(filtered_state_means[:,0],filtered_state_means[:,1],'b',label='kalman output')
#plt.hold(True)
plt.legend(loc=3)
plt.title("Constant Velocity Kalman Filter")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()