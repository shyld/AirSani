
import pandas as pd
import numpy as np
import time
import math
import datetime 
import steer
#from azure.iot.device import IoTHubDeviceClient, Message


df = pd.read_csv('arguments.csv')
# read arguments
F_running = (df[df['arguments']=='F3']['value'].iloc[0]=='TRUE') & (df[df['arguments']=='F0']['value'].iloc[0]=='TRUE')
Cam_width = int(df[df['arguments']=='Cam_width']['value'].iloc[0])
Cam_height= int(df[df['arguments']=='Cam_height']['value'].iloc[0])
Check_stop_period = int(df[df['arguments']=='Check_stop']['value'].iloc[0])
Coverage_size_x = int(df[df['arguments']=='Coverage_size']['value'].iloc[0])
UV_period = int(df[df['arguments']=='UV_period']['value'].iloc[0]) # Min Exposure Time for each spot
Num_UV_LED = int(df[df['arguments']=='Num_UV_LED']['value'].iloc[0])
UV_after_sec = int(df[df['arguments']=='UV_after_sec']['value'].iloc[0])
Cloud_seq_len = int(df[df['arguments']=='Cloud_seq_len']['value'].iloc[0])
UV = df[df['arguments']=='UV']['value'].iloc[0]=='TRUE'

i0=5
j0=5


WaitTime = .0005  # motor speed 
Coverage_size_y = Coverage_size_x

df_empty = pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})

sanitization_list=[]
##

#Cumulative sanitization matrix
C = np.zeros((2*i0,2*j0))

# update check error file
t = datetime.datetime.now()
df = pd.DataFrame({'F1':[t], 'F2':[t], 'F3':[t]})
df.to_csv('check_error.csv',index=False)


#######################
########################  Motor Functions ###############

# load UV_coordinates from scored_spots 
def get_UV_coordinate():
    #df_priority = pd.read_csv('unsorted_scores.csv')
    # covert to correct data types
    #df_priority['time'] = pd.to_datetime(df_priority['time'])

    df = shared_variables.scored_spots
    # sort and filter df_priority
    df_sorted = df[df['time']< datetime.datetime.now() - datetime.timedelta(seconds=UV_after_sec)]

    if len(df_sorted)>0:
        df_sorted = df_sorted.sort_values(['priority','score'], ascending=False)
        df_top_locations = df_sorted.iloc[0:Num_UV_LED,2:4] 
        # save to csv (UV_priority.csv)
        #df_sorted.to_csv('UV_priority.csv',index=False)
    return df_top_locations


# update scores in unsorted_scores.csv
def update_scores(df_top_locations):

    # update df_priority
    for k in len(df_top_locations):
        i,j = df_top_locations.iloc[k,0],df_top_locations.iloc[k,1]
        ## update the scores
        idx  = (shared_variables.scored_spots['i']==i) & (shared_variables.scored_spots['j']==j)
        shared_variables.scored_spots.loc[idx,'score'] -= 1 
        ## remove zero scores
        shared_variables.scored_spots = shared_variables.scored_spots[shared_variables.scored_spots.score != 0].reset_index(drop=True)
        if len(shared_variables.scored_spots)==0:
            shared_variables.scored_spots = df_empty
    #print('MMMMMMMM  ', df_priority.columns)
    #df_priority.to_csv('unsorted_scores.csv',index=False)

# A function to check the stopping rule
def check_stop(b0,b1):
    F_running = True
    if b1 != b0:
        print('Check_stop')
        b0 = b1
        df0 = pd.read_csv('config.csv')
        F_running = (df0[df0['arguments']=='F3']['value'].iloc[0]=='TRUE') & (df0[df0['arguments']=='F0']['value'].iloc[0]=='TRUE')
    return b0, F_running


#######################
######################## main code ###############

a0,a1,b0,b1 = 0,0,0,0
d1_prev,d2_prev = 0,0

# setup the output pins
if UV:
    steer.setup_pins()



# process frames until user exits
while F_running:

    t = datetime.datetime.now()
    a1 =  int(t.microsecond/(UV_period))
    b1 =  int(t.second/Check_stop_period)

	# Check_stop
    b0, F_running = check_stop(b0,b1)

    if a1 != a0:
        a0 = a1

        try:
            top_locations = get_UV_coordinate()
        except:
            continue

        # update scores based on the current sanitization locations
        if len(top_locations)>0:
            steer.apply_UV(top_locations)
            update_scores(top_locations)




    # update check_error file
    df_check = pd.read_csv('check_error.csv')
    df_check['F3'] = pd.to_datetime(df_check['F3'])
    df_check['F3'] = [t]
    df_check.to_csv('check_error.csv',index=False)
