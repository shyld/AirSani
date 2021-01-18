import pandas as pd
import numpy as np
import time
import math
import datetime 



def closest_point(point, points):
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return np.argmin(dist_2)


# This function assigns UV Engines to Locations that Need to be Sanitized
# Function input is UV_status, a 4*1 list. 0 means UV is not available, 1 means UV is idle
# shared_variables.assigned_UV for idle UVs is updated in this function

# scored_spots is a shared variable that has 5 columns (time,priority,x,y,score)
# UV_current_xy is a shared variable that is 2*4 and stores the last location assigned to each UV Engine


def assign_UV(UV_status)

    #if all(uv == 0 for uv in UV_status): # Check if all UVs are active, i.e. UV_status = [0,0,0,0]


    #idle_UVs = np.nonzero(UV_status) # get indices of idle UVs

    x = shared_variables.scored_spots[:,2] # x coordinates from scored_spots
    y = shared_variables.scored_spots[:,3] # y coordinates from scored_spots
    # Assign the spot to UV Engine 1
    if UV_status[0]==1:
        mask = np.logical_and(np.logical_and(x<400, y<400),np.logical_and((x-UV_current_xy[0,1])**2+(y-UV_current_xy[1,1])**2>1600,(x-UV_current_xy[0,2])**2+(y-UV_current_xy[1,2])**2>1600))
        A = shared_variables.scored_spots[mask,:]
        indx = numpy.where(A[:,4] == numpy.amax(A[:,4]))
        shared_variables.UV_current_xy[0] = A[:,indx] #Assign the element in A with highest score to UV Engine 1
        A = np.delete(A, indx, 0) #delete the assigned xy from scored_spots

    # Assign the spot to UV Engine 2
    if UV_status[1]==1:
        mask = np.logical_and(np.logical_and(x>-400, y<400),np.logical_and((x-UV_current_xy[0,0])**2+(y-UV_current_xy[1,0])**2>1600,(x-UV_current_xy[0,2])**2+(y-UV_current_xy[1,2])**2>1600))
        A = shared_variables.scored_spots[mask,:]
        indx = numpy.where(A[:,4] == numpy.amax(A[:,4]))
        shared_variables.UV_current_xy[1] = A[:,indx] #Assign the element in A with highest score to UV Engine 2
        A = np.delete(A, indx, 0) #delete the assigned xy from scored_spots

    # Assign the spot to UV Engine 3
    if UV_status[2]==1:
        mask = np.logical_and(np.logical_and(y<1200, y>-400),np.logical_and((x-UV_current_xy[0,1])**2+(y-UV_current_xy[1,1])**2>1600,(x-UV_current_xy[0,2])**2+(y-UV_current_xy[1,2])**2>1600))
        A = shared_variables.scored_spots[mask,:]
        indx = numpy.where(A[:,4] == numpy.amax(A[:,4]))
        shared_variables.UV_current_xy[2] = A[:,indx] #Assign the element in A with highest score to UV Engine 3
        A = np.delete(A, indx, 0) #delete the assigned xy from scored_spots


        
        
    
    
    
