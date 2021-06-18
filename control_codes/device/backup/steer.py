import pandas as pd
import numpy as np
import time
import math
import datetime 
import RPi.GPIO as GPIO
import argparse 


WaitTime = .0005
theta=[0,0,0,0]
degree_steps = 64/5.62/2
pixel_steps = (64/5.62/2)/13.8

path = '/home/shyldai/shyld/AirSani/control_codes/device/'


# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.")
parser.add_argument("--LED", type=int, default=1, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--x", type=int, default=0, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--y", type=int, default=0, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--light", type=int, default=0, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("error parsing")
	parser.print_help()
	sys.exit(0)

PATH = os.path.dirname(os.path.abspath(__file__))

#################################################
	## update check_error file


class UV:
	
	def setup_pins(self):
		s = 8
		GPIO.cleanup()
		GPIO.setmode(GPIO.BCM)
		GPIO.setup(27,GPIO.OUT)
		GPIO.setup(26,GPIO.OUT)
		GPIO.setup(25,GPIO.OUT)
		GPIO.setup(24,GPIO.OUT)
		GPIO.setup(23,GPIO.OUT)
		GPIO.setup(22,GPIO.OUT)
		GPIO.setup(21,GPIO.OUT)
		GPIO.setup(20,GPIO.OUT)
		GPIO.setup(19,GPIO.OUT)
		GPIO.setup(18,GPIO.OUT)
		GPIO.setup(17,GPIO.OUT)
		GPIO.setup(16,GPIO.OUT)
		GPIO.setup(13,GPIO.OUT)
		GPIO.setup(12,GPIO.OUT)
		GPIO.setup(11,GPIO.OUT)
		GPIO.setup(10,GPIO.OUT)
		GPIO.setup(9,GPIO.OUT)
		GPIO.setup(8,GPIO.OUT)
		GPIO.setup(6,GPIO.OUT)
		GPIO.setup(7,GPIO.OUT)
		GPIO.setup(5,GPIO.OUT)
		GPIO.output(s, False)

	def __init__(self):

		# setup pins
		self.setup_pins()

		# initilize variables
		self.phi0= np.load('UV_coordinates_phi.npy')
		self.rho0= np.load('UV_coordinates_rho.npy')
		self.x0 = 0
		self.y0 = 0
		
		self.light_on =[False, False, False, False]

		# Initialize the locations

		# A function to initialize only one LED, keeping the other LEDs off



	def convert_xy2degree(self,x,y,LED): # -width/2 < x < width/2

		#x1,y1=rotate(x,y)
		# the rotation is so that the safely rotating area is towards positive x (negative x is prohibited)
		print(WaitTime)
		#print(self.rho0,self.phi0)

		rho = np.sqrt(x**2+y**2)
		phi = np.arctan2(y,x)/np.pi*180
		print('rho,phi: ', rho,phi)
		phi_bias = np.zeros(4)
		phi_bias[0] = 0
		phi_bias[1] = 0
		phi_bias[2] = -180
		phi_bias[3] = -180

		phi_t = phi - phi_bias[LED]
	        
		if phi_t<0:
			rho = -rho
			phi_t = phi_t +180
			
		if phi_t>180:
			rho = -rho
			phi_t = phi_t -180 

		#phi0 = np.load(path+'UV_coordinates_phi.npy')
		#rho0 = np.load(path+'UV_coordinates_rho.npy')
		print('rho0,phi0: ', self.rho0,self.phi0)

		delta_rho = rho - self.rho0[LED]
		delta_phi = phi_t - self.phi0[LED]

		self.rho0[LED] = rho
		self.phi0[LED] = phi_t

		#np.save(path+'UV_coordinates_phi', self.phi0)
		#np.save(path+'UV_coordinates_rho', self.rho0)

		# apply limits
		d1 = -1*int(pixel_steps * delta_rho)
		d2 = int(degree_steps * delta_phi)

		print('d1,d2: ', d1,d2)
		print('delta_phi: ', delta_phi)
		print('self.phi0[LED]: ', self.phi0[LED])

		return d1, d2, rho, phi_t

	def update_light_locations(self,x,y,LED):
		d1, d2, rho, phi_t = convert_xy2degree(self,x,y,LED)
		self.rho0[LED] = rho
		self.phi0[LED] = phi_t


	# Motor and driver funtions
	def motor(self,d,p1,p2):
	    #n = int(abs(d)/ (2*5.625 * 1/64))
	    #print('motor: ', d)
		if d>0:
			A = False
		else:
			A = True

	    #print(p1,A)
		GPIO.output(p1, A)

		for i in range(np.abs(d)):        
			GPIO.output(p2, False)
			time.sleep(WaitTime)
			GPIO.output(p2, True)
			time.sleep(WaitTime)


	def UV_driver(self,x1,y1,uv,p1,p2,q1,q2,l,LED):
		
		x,y,_,_=self.convert_xy2degree(x1,y1,LED)

		s = 8 # This pin Puts Motors in Sleep or active
		GPIO.output(s, True)
		# turn off the light before any movement
		GPIO.output(l,False)

		if x!=0 or y !=0:
			
			self.motor(x,p1,p2)
			self.motor(y,q1,q2)	

		if uv:
			GPIO.output(l,True)
		else:
			GPIO.output(l,False)

		GPIO.output(s, False)


	def get_pins(self,LED_id):
		if LED_id==0:
			p1,p2 = 25,24 # direction, pulse 
			q1,q2 = 16,12 # direction, pulse 
			l = 7
		elif LED_id==1:
			p1,p2 = 21, 20
			q1,q2 = 26,19
			l = 11
		elif LED_id==2:
			p1,p2 = 6,13
			q1,q2 = 5, 22
			l = 10   
		elif LED_id==3:
			p1,p2 = 17,27
			q1,q2 = 18, 23
			l = 9  
		return p1,p2,q1,q2,l


	def UV_go(self,LED,x,y,light_on): # LED \in {0,1,2,3}
		x = int(x)
		y = int(y)

		p1,p2,q1,q2,l = self.get_pins(LED_id = LED)
		self.UV_driver(x1= x, y1=y, uv=light_on,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=LED)
		print(LED,x,y,light_on)


#### Functions for two step steering

	def find_nearest_loc(self, x=0,y=0):

		df_loc = pd.read_csv(PATH+'/light_loc_from_IR.csv')
		
		centers = []# read from CSV
		for i in range(len(df_loc)):
			centers,append(df_loc.iloc[i,:].tolist())

		if centers.shape[0]>0:
			# find the nearest light
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centers)
			distances, indices = nbrs.kneighbors([[x,y]])
			#print(distances)
			# if the light is close
			neighborhood = 50
			if distances[0,0]<neighborhood:
				return centers[indices[0,0],:]
			else:
				return None

	def report_light_status(self, LED, light_status):
		# write in to the CSV file
		PATH = os.path.dirname(os.path.abspath(__file__))
		try:
			df_status = pd.read_csv(PATH+'/light_status.csv')
		except:

			df_status = pd.DataFrame({'LED0': [False],'LED1': [False],'LED2': [False],'LED3': [False]})

		df_check['LED'+str(LED)] = [light_status]
		#df_check[s] = [datetime.datetime.now()]
		df_check.to_csv(PATH+'/light_status.csv',index=False)


	def two_step_steer(self, LED, x,y): # the light will automatically turn on
		x = int(x)
		y = int(y)
		
		# Step 1: Turn off light (without changing the location)
		self.UV_go(LED,self.x0,self.y0,light_on=False)
		report_light_status(LED, False)

		time.sleep(0.5)
		# Step 2: Move to the new location and then turn the light on
		self.UV_go(LED,x,y,light_on=True)
		report_light_status(LED, True)
		# Update the current position of LEDs
		self.update_light_locations(x,y,LED)

		# check if the new light location is far from the center, otherwise don't need two-step steering
		if (x**2 + y**2)<100**2:
			#self.x0 = x
			#self.y0 = y
			# Update the current position of LEDs
			#self.update_light_locations(x,y,LED)
			return 0

		time.sleep(0.5)
		try:
			[x_new,y_new] = self.find_nearest_loc(x,y)

			#self.x0 = x_new
			#self.y0 = y_new
			# Update the current position of LEDs
			self.update_light_locations(x_new,y_new,LED)
			self.UV_go(LED, x, y,light_on=True)

		except:
			print('Only one step steering happened')












if __name__ =='__main__':

	# initialize:
	setup_pins()

	rho0,phi0=np.zeros(4),np.zeros(4)
	_,_,a1,a2 = convert_xy2degree(x=242,y=92,LED=0)
	rho0[0],pho0[0] = a1,a2


	# Test
	p1,p2,q1,q2,l = get_pins(LED_id = 0)
	#UV_driver(x1= opt.x, y1=opt.y, uv=uv,p1=p1,p2=p2,q1=q1,q2=q2,l=l)
	w = 5

	print('Turning on LED 0:')
	x,y = 0,0
	UV_driver(x1= x, y1=y, uv=False,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=1)
	#time.sleep(w)
	input("Press Enter to continue...")

	print('50,0')
	x,y = 1000,0
	UV_driver(x1= x, y1=y, uv=False,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=1)
	#time.sleep(w)
	input("Press Enter to continue...")

	print('100,0')
	x,y = 800,0
	UV_driver(x1= x, y1=y, uv=False,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=1)
	#time.sleep(w)
	input("Press Enter to continue...")

	print('-50,0')
	x,y = -500,0
	UV_driver(x1= x, y1=y, uv=False,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=1)
	#time.sleep(w)
	input("Press Enter to continue...")

	print('-100,0')
	x,y = -1000,0
	UV_driver(x1= x, y1=y, uv=False,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=1)
	#time.sleep(w)
	input("Press Enter to continue...")


	print('0,50')
	x,y = 0,500
	UV_driver(x1= x, y1=y, uv=False,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=1)
	#time.sleep(w)
	input("Press Enter to continue...")
	
	print('0,-50')
	x,y = 0,-500
	UV_driver(x1= x, y1=y, uv=False,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=1)
	input("Press Enter to continue...")
