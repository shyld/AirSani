
import numpy as np
import os
import pandas as pd

def read_light_status():
    # write in to the CSV file
    PATH = os.path.dirname(os.path.abspath(__file__)) # path of the current file (not the master file)
    if True:
        #print(PATH+'/device/light_status.csv')
        df_status = pd.read_csv(PATH+'/light_status.csv')
        df_status = df_status[['LED0','LED1','LED2','LED3']].astype('str')
    if False:

        df_status = pd.DataFrame({'LED0': [False],'LED1': [False],'LED2': [False],'LED3': [False]})

    L = df_status.iloc[0,:].tolist()
    #print('L', L)
    return L


def write_light_loc_from_IR(centers):
    PATH = os.path.dirname(os.path.abspath(__file__))    
    if len(centers[0])==0:
        df_status = pd.DataFrame({'X': [],'Y': []})
        print(PATH+'/light_loc_from_IR.csv')
        df_status.to_csv(PATH+'/light_loc_from_IR.csv',index=False)
        return df_status

print(read_light_status())