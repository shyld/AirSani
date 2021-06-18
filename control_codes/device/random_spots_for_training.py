from steer import UV
import os
import pandas as pd
import time 
from UV_coordinate_initialize import init_coordinates
import numpy as np

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


# Initialize
LOC_MAP_list = []
LED_id = 0
my_UV = UV()
my_UV.UV_all_off()
time.sleep(3)
x,y = my_UV.find_loc(LED=LED_id)
my_UV.one_step_steer(LED=LED_id , x=0,y=0)
my_UV.one_step_steer(LED=LED_id , x=0,y=0)
x,y = my_UV.find_loc(LED=LED_id)
my_UV.one_step_steer(LED=LED_id , x=0,y=0)
print(x,y)

x_low = -150
x_high = 150
y_low = -150
y_high = 150

#my_UV.one_step_steer(LED=LED_id , x=0,y=0)

for i in range(10):
	x = np.random.randint(low=x_low,high=x_high,size=1)[0]
	y = np.random.randint(low=y_low,high=y_high,size=1)[0]
	print('*** moving to:\n\n\n\n\n', x,y)
	print('moving to:', x,y)
	
	my_UV.one_step_steer(LED_id,x,y,light_on=True)
	x_m,y_m = my_UV.read_recent_loc(LED=LED_id)
	time.sleep(1)
	my_UV.one_step_steer(LED_id,0,0,light_on=True)
	time.sleep(1)
	x_d,y_d = my_UV.read_recent_loc(LED=LED_id)
	my_UV.one_step_steer(LED_id,0,0,light_on=True)

	LOC_MAP_list.append([x,y,x_m,y_m,x_d,y_d])
	#my_UV.one_step_steer(LED=LED_id , x=x,y=y)
my_UV.UV_all_off()

print(LOC_MAP_list)
LOC_MAP = np.array(LOC_MAP_list)
np.save('LOC_MAP.npy', LOC_MAP)
