
import math
import pandas as pd
import numpy as np
import datetime 
#from control_codes import shared_variables
import shared_variables
import os
import time 
from device.steer import UV #control_codes.
#from azure.iot.device import IoTHubDeviceClient, Message
shared_variables.init()

############################# PARAMTERS ###############
#shared_variables.init()
PATH = os.path.dirname(os.path.abspath(__file__))
path = PATH
############################

LED_id = 1

my_UV = UV()


my_UV.UV_LED_on(LED_id)
print('LED_id: ',LED_id)
time.sleep(2)
my_UV.UV_LED_off(LED_id)
my_UV.reset_loc(LED=LED_id)

#time.sleep(2)
#LED = 0
#x,y = my_UV.init_loc(LED)
#print('LED initialized. location x,y', x,y)


#################################################
    ## update check_error file
def set_log(s):
    PATH = os.path.dirname(os.path.abspath(__file__))
    df_check = pd.read_csv(PATH+'/shared_csv_files/log.csv')
    df_check[s] = pd.to_datetime(df_check[s])
    df_check[s] = [datetime.datetime.now()]
    df_check.to_csv(PATH+'/shared_csv_files/log.csv',index=False)

set_log('F3')


def LED_in_area(LED=0,x=0,y=0):
    if LED==0 or LED==1 or LED==2 or LED==3:
        if -100<x and x<100 and -100<y and y<100:
            return True
        else:
            return False

#shared_variables.init()
print('opened sanitize.py')


a0,a1, b0, b1, c0,c1 = 0,0,0,0,0,0
#parsing_time = datetime.datetime(2010, 1, 1, 1, 1)

# update check error file
t = datetime.datetime.now()

# A function to check the stopping rule
def check_stop(b0,b1):
    F_running = True
    if b1 != b0:
        #print('Check_stop')
        b0 = b1
        df0 = pd.read_csv(path+'/shared_csv_files/onoff.csv')
        F_running = (df0[df0['arguments']=='F3']['value'].iloc[0]=='TRUE') & (df0[df0['arguments']=='F0']['value'].iloc[0]=='TRUE')
    return b0, F_running

F_running = True

# process frames until user exits
#while True:
x_prev = 0
y_prev = 0
while F_running:
    time.sleep(0.3)
    t = datetime.datetime.now()
    b1 =  int(t.second)
    #b0, F_running = check_stop(b0,b1)

    try:
        df_score = pd.read_csv(PATH+'/shared_csv_files/scored_spots.csv')
        #df_new =  pd.concat([df, detected_coordinates])
    except:
        df_score = pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})
        print('NOT Read FROM scored_spots')


    # covert to correct data types
    if len(df_score)>0:
        df_score['time'] = pd.to_datetime(df_score['time'])
    else:
        continue
    #print(shared_variables.scored_spots)
    #print('*****')
    #print(shared_variables.scored_spots['score']==-1)
    idx = (df_score['score']==-1) & (df_score['priority'] == LED_id)
    df_UV = df_score[idx]

    #print(df_score[df_score['score']==-1])

    #print('df_score: ', df_score)
    if len(df_UV)>0:
        print('df_UV: ', df_UV)
        #print('111111111     df_UV len', len(df_UV))
        # remove the indexes
        
        #df_score  = df_score[idx].drop().reset_index(drop=True)
        #df_score.to_csv(PATH+'/shared_csv_files/scored_spots.csv',index=False)

        # plot 
        #for i in range(len(df_UV)):
        i = 0

        t,pr,x1,y1,sc = df_UV['time'].iloc[i], int(df_UV['priority'].iloc[i]), int(df_UV['i'].iloc[i]), int(df_UV['j'].iloc[i]), int(df_UV['score'].iloc[i])
        #print('UV: x,y ',x1,y1)

        # Apply UV
        print('x1,x_prev ,y1,y_prev', x1,x_prev ,y1,y_prev)
        if (x1 != x_prev or y1 !=y_prev): #and LED_in_area(LED=LED_id,x =x1,y=y1):
            x_prev = x1
            y_prev = y1
            
            x1 = x1 + shared_variables.Coverage_size/2
            y1 = -(y1 + shared_variables.Coverage_size/2)
            print('Sanitize: Two Step Steering to x,y ', x1,y1)
            #my_UV.one_step_steer(LED=LED, x=x1,y=y1)
            #x,y = my_UV.init_loc(LED=1)
            #my_UV.UV_go(LED=1,x=x1,y=y1,light_on=True)
            #x,y = my_UV.find_loc(LED=LED_id)
            my_UV.one_step_steer(LED=LED_id, x=x1,y=y1)
    else:
        my_UV.UV_LED_off(LED_id)



my_UV.UV_all_off()

