from steer import UV
import os
import pandas as pd
import time 
from UV_coordinate_initialize import init_coordinates

def read_light_status():
	# write in to the CSV file
	PATH = os.path.dirname(os.path.abspath(__file__))
	try:
		df_status = pd.read_csv(PATH+'/light_status.csv')
	except:

		df_status = pd.DataFrame({'LED0': [False],'LED1': [False],'LED2': [False],'LED3': [False]})

	return df_status

def write_light_loc_from_IR(x,y):
	# write in to the CSV file
	
	#df_loc = pd.read_csv(PATH+'/light_loc_from_IR.csv')
	
	PATH = os.path.dirname(os.path.abspath(__file__))

	df_status = pd.DataFrame({'X': [x],'Y': [y]})
	#df_check['LED'+str(LED)] = [light_status]
	#df_check[s] = [datetime.datetime.now()]
	df_status.to_csv(PATH+'/light_loc_from_IR.csv',index=False)


#init_coordinates() 
#write_light_loc_from_IR(150,150)
i=0
my_UV = UV()
my_UV.reset_loc(LED=i)
time.sleep(5)
my_UV.one_step_steer(LED=i, x=50,y=50)
time.sleep(5)
my_UV.reset_loc(LED=i)
#my_UV.UV_go(LED=0,x=0,y=-50,light_on = True)

#x,y = my_UV.init_loc(LED)
#my_UV.UV_go(LED,30,30,light_on=True)
#time.sleep(5)
#my_UV.UV_go(LED,0,0,light_on=False)