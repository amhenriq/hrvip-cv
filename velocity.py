from pandas import DataFrame
import pandas as pd 
import numpy as np
file = open('datafile2.csv', "w")
df=pd.read_csv('datafile.csv', header=None, delim_whitespace=True)
#use commas instead of spaces

df = DataFrame(df)

df.columns = ['Time', 'X', 'Y', 'Z'] #renames 0,1,2 to time, xcord, ycord
dist = df.diff().fillna(0)
Distance = np.sqrt(dist.X**2 + dist.Y**2)
Timerate = dist.Time
Velocity = Distance / Timerate

df = df.assign(Distance=pd.Series(Distance))
df = df.assign(Velocity=pd.Series(Velocity)).fillna(0)

df.to_csv('datafile2.csv', sep='\t')
#accelerations
df2=pd.read_csv('datafile2.csv', header=None, delim_whitespace=True, skiprows=1)
df2.columns = ['blank','Time', 'X', 'Y', 'Z','Distance','Velocity']
dist2 = df2.diff().fillna(0)
VelocityDiff=dist2.Velocity
Acceleration = VelocityDiff / Timerate
df2 = df2.assign(Acceleration=pd.Series(Acceleration)).fillna(0)


file = open('acceleration.csv', "w")
header = ["Time", "Distance", "Velocity", "Acceleration"]
df2.to_csv('acceleration.csv', columns=header, sep='\t')

#########################################################
df = df.replace(0, np.NaN) #replaces all zero values with NaN for plotting 
#df.plot.line()
df.plot(x='Time', y=['X','Y','Distance','Velocity'])
df2 = df2.replace(0, np.NaN)
#df2.plot(x='Time', y=['Velocity','Acceleration'])
df2.plot(x='X', y='Y')
df2.title("Float Unit Position")
df2.xlabel('X')
df2.ylabel('Y')
#########################################################
print(df2)