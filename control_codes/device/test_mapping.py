import pandas as pd
import numpy as np
import time
import math
import sklearn
from sklearn.neighbors import NearestNeighbors

# load file
LOC_MAP = np.load('LOC_MAP.npy')
LOC_MAP = LOC_MAP[:,0:4].astype(int)


x = 60
y = 45

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(LOC_MAP[:,2:4])
distances, indices = nbrs.kneighbors([[x,y]])

print(indices)
print(LOC_MAP[indices[0,0],:])
[x_motor_0, y_motor_0, x_IR_0, y_IR_0] =  LOC_MAP[indices[0,0],:]

print(x_motor_0,y_motor_0)
x_final = int(x/x_IR_0 * x_motor_0)
y_final = int(y/y_IR_0 * y_motor_0)
print(x_final,y_final)
#d1, d2, X, Y = self.convert_xy2degree(x,y,LED)

