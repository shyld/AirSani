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


init_coordinates() 
#write_light_loc_from_IR(150,150)
my_UV = UV()

# light_off
my_UV.UV_all_off()
# light_on
#my_UV.UV_all_on()
#time.sleep(2)
#print(my_UV.find_nearest_loc(x=120,y=120))
# light_off
#my_UV.UV_all_off()

if True:
	#time.sleep(2)
	# move to 100, 100
	#print('my_UV.rho0[LED],my_UV.phi0[LED] ', my_UV.rho0,my_UV.phi0)
	for i in range(4):
		LED =i
		my_UV.UV_LED_on(LED)
		print(i)
		time.sleep(2)
		my_UV.UV_LED_off(LED)
        
      
	my_UV.UV_all_off()
	time.sleep(2)
	#my_UV.reset_loc(LED)
	LED = 0
	x,y = my_UV.init_loc(LED)
	print('LED initialized. location x,y', x,y)
	#time.sleep(5)
	#my_UV.one_step_steer(LED=0, x=int(x/2),y=int(y/2))
	time.sleep(5)	
	print('moving to x,y', 20,40)
	print('my_UV.rho0[LED],my_UV.phi0[LED] ', my_UV.rho0,my_UV.phi0)
	#my_UV.one_step_steer(LED=LED, x=100,y=100)
	my_UV.two_step_steer(LED=LED, x=20,y=40)
	#my_UV.reset_loc(LED)
	#print('moving to', 0,0)
	#my_UV.one_step_steer(LED=0, x=0,y=0)
	#my_UV.one_step_steer(LED=0, x=100,y=100)
	print('my_UV.rho0[LED],my_UV.phi0[LED] ', my_UV.rho0,my_UV.phi0)
	df = read_light_status()
	print(df)
	#print('first_round_done')
	#time.sleep(5)

	# go back to the origin
	#my_UV.two_step_steer(LED=0, x=150,y=0)
	#print('my_UV.rho0[LED],my_UV.phi0[LED] ', my_UV.rho0,my_UV.phi0)
	#df = read_light_status()
	#print(df)

# light_off
#my_UV.reset_loc(LED)
my_UV.UV_all_off()
my_UV.report_light_status(0, 'off')
print('end')