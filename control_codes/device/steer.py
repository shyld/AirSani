import pandas as pd
import numpy as np
import time
import math
import datetime 
import RPi.GPIO as GPIO
import argparse 
import os
import sklearn
from sklearn.neighbors import NearestNeighbors

WaitTime = .0007 #.0005
theta=[0,0,0,0]

degree_steps = 64/5.62/2 *1.2
pixel_steps = (64/5.62/2)/13.8*5.5*0.8
wait_before_photo = 1 # sec

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
		#GPIO.setup(14,GPIO.OUT)
		GPIO.setup(5,GPIO.OUT)
		GPIO.output(s, False)
		#GPIO.output(14, True)

	def __init__(self):

		# setup pins
		self.setup_pins()

		# initilize variables
		#self.X0= np.load(path+'UV_coordinates_phi.npy')
		#self.Y0= np.load(path+'UV_coordinates_rho.npy')
		self.Alpha0 = [0,0,0,0]
		self.Beta0 = [0,0,0,0]
		
		self.light_on =[False, False, False, False]

		# load location mapping
		# load file
		PATH = os.path.dirname(os.path.abspath(__file__))
		LOC_MAP = np.load(PATH + '/LOC_MAP.npy')
		self.LOC_MAP = LOC_MAP[:,0:4].astype(int)



	def convert_xy2degree(self,x,y,LED): # -width/2 < x < width/2


		# transfor x,y to alpha and beta based on non-linearity
		rho = np.sqrt(x**2+y**2) * pixel_steps/degree_steps * np.pi/180
		phi = np.arctan2(abs(y),abs(x))

		print('rho,phi: ', rho,phi)

		Alpha = math.atan(math.cos(phi)*math.tan(rho))
		Beta = math.acos(math.cos(rho)/math.cos(Alpha))
		print('Alpha,Beta: ', Alpha,Beta)

		if x < 0:
			Alpha = -Alpha
		if y < 0:
			Beta = - Beta


		delta_Alpha = Alpha - self.Alpha0[LED]
		delta_Beta = Beta - self.Beta0[LED]
		print('delta_Alpha, delta_Beta: ', delta_Alpha, delta_Beta)
		self.Alpha0[LED] = Alpha
		self.Beta0[LED] = Beta

		# adjust steps based on polar param
		d1 = int(degree_steps * delta_Alpha/np.pi*180)  #(-1 for direction)
		d2 = int(degree_steps * delta_Beta/np.pi*180 )  

		if LED==0:
			d1 = -d1
			d2 = -d2 
		elif LED ==1:
			d1 = d1
			d2 = -d2
		elif LED ==2:
			d1 = d1
			d2 = d2
		elif LED ==3:
			d1  = d2
			d2 = d1

		print('d1,d2: ', d1,d2)

		return d1, d2, Alpha, Beta

	def update_light_locations(self,x,y,LED):
		d1, d2, X, Y = self.convert_xy2degree(x,y,LED)
		#print()
		self.Alpha0[LED] = X
		self.Beta0[LED] = Y


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
		
		d1,d2,_,_=self.convert_xy2degree(x1,y1,LED)

		s = 8 # This pin Puts Motors in Sleep or active
		GPIO.output(s, True)
		# turn off the light before any movement
		GPIO.output(l,False)

		if d1!=0 or d2 !=0:
			
			self.motor(d1,p1,p2)
			self.motor(d2,q1,q2)	

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
			q1,q2 = 18, 9
			l = 23  
		return p1,p2,q1,q2,l


	def UV_go(self,LED,x,y,light_on): # LED \in {0,1,2,3}
		x = int(x)
		y = int(y)

		p1,p2,q1,q2,l = self.get_pins(LED_id = LED)
		self.UV_driver(x1= x, y1=y, uv=light_on,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=LED)
		#print('UV_go',LED,x,y,light_on)


	def UV_all_off(self): # LED \in {0,1,2,3}
		for i in range(4):
			_,_,_,_,l = self.get_pins(i)
			GPIO.output(l,False)

	def UV_LED_off(self, LED): # LED \in {0,1,2,3}
		_,_,_,_,l = self.get_pins(LED)
		GPIO.output(l,False)

	def UV_LED_on(self, LED): # LED \in {0,1,2,3}
		_,_,_,_,l = self.get_pins(LED)
		GPIO.output(l,True)

	def UV_all_on(self): # LED \in {0,1,2,3}
		for i in range(4):
			_,_,_,_,l = self.get_pins(i)
			GPIO.output(l,True)

#### Functions for two step steering
	def read_IR_locations(self):

		df_loc = pd.read_csv(PATH+'/light_loc_from_IR.csv')
		
		centers = []# read from CSV
		for i in range(len(df_loc)):
			centers.append(df_loc.iloc[i,:].tolist())
		
		centers = np.array(centers)
		return centers

	def find_nearest_loc(self, x=0,y=0):

		centers = self.read_IR_locations()
		if centers.shape[0]>0:
			# find the nearest light
			#print('find_nearest_loc, x,y,centers ', x,y, centers )
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centers)
			distances, indices = nbrs.kneighbors([[x,y]])
			#print(distances)
			# if the light is close
			neighborhood = 30
			if distances[0,0]<neighborhood:
				#print(centers)
				return centers[indices[0,0]]
			else:
				return None

	def map_locations(self, x=0,y=0):
		nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.LOC_MAP[:,2:4])
		distances, indices = nbrs.kneighbors([[x,y]])

		#print(indices)
		#print(LOC_MAP[indices[0,0],:])
		[x_motor_0, y_motor_0, x_IR_0, y_IR_0] =  self.LOC_MAP[indices[0,0],:]

		#print(x_motor_0,y_motor_0)
		x_final = int(x/x_IR_0 * x_motor_0)
		y_final = int(y/y_IR_0 * y_motor_0)
		#print(x_final,y_final)
		return x_final, y_final



	def report_light_status(self, LED, light_status, x,y):
		# write in to the CSV file
		PATH = os.path.dirname(os.path.abspath(__file__))
		try:
			df_status = pd.read_csv(PATH+'/light_status.csv')
		except:
			print('ERROR REPORTING LIGHT STATUS')
			df_status = pd.DataFrame({'LED0': [False,0,0],'LED1': [False,0,0],'LED2': [False,0,0],'LED3': [False,0,0]})

		df_status['LED'+str(LED)] = [light_status,x,y]
		#df_check[s] = [datetime.datetime.now()]
		df_status.to_csv(PATH+'/light_status.csv',index=False)


	def one_step_steer(self, LED, x,y,light_on=True): # one step steer with IR positioning
		if LED==3:
			x1= x
			x = y
			y = x1
		#print('^^^ One Step starting: x,y', x,y)
		x_new, y_new = self.map_locations(x,y)
		print('mapped x y', x_new, y_new)
		
		x = x_new
		y = y_new

		self.UV_LED_off(LED)
		self.report_light_status(LED, 'off',x,y)
		time.sleep(wait_before_photo)
		#print('Step 1: IR is off. Now can take frame for background')
		# Step 1: Move to the new location and then turn the light on
		self.report_light_status(LED, 'transition',x,y)
		self.UV_go(LED,x,y,light_on=False)
		#time.sleep(3)
		
		#time.sleep(wait_before_photo)
		self.UV_LED_on(LED)
		#print('LED on')
		time.sleep(wait_before_photo) # to be removed in the product code
		self.report_light_status(LED, 'on',x,y)
		time.sleep(wait_before_photo)
		#time.sleep(0.5)
		# Update the current position of LEDs (x,y) in the internal class variables
		x_new,y_new = self.read_recent_loc(LED=LED)
		self.update_light_locations(x_new,y_new,LED)

		if light_on==False:
			self.UV_LED_off(LED)
			self.report_light_status(LED, 'off',x,y)
		#print('report_light_status(LED, True)')
		#print('self.rho0[LED],self.phi0[LED] ', self.rho0[LED],self.phi0[LED])
		#print('*** one step ended ***')


	def two_step_steer(self, LED, x,y): # the light will automatically turn on
		
		self.one_step_steer(LED, x,y)
		print('One Step Steering Done')
		# check if the new light location is far from the center, otherwise don't need two-step steering
		#if (x**2 + y**2)<20**2:
		#	print('light in the center area')
		#	return 0
		#print('waiting for 1 sec before taking step 2')
		#time.sleep(wait_before_photo)
		try:
			#print('Finding the neast neighbor locations in the cam')         
			[x_new,y_new] = self.find_nearest_loc(x,y)
			print('x_new,y_new', x_new,y_new)

			#self.x0 = x_new
			#self.y0 = y_new
			# Update the current position of LEDs
			self.update_light_locations(x_new,y_new,LED)
			#self.UV_go(LED, x, y,light_on=True)
			self.one_step_steer(LED, x,y)

			print('Second Steering Done: ')
			[x_new,y_new] = self.find_nearest_loc(x,y)
			print('x_new,y_new', x_new,y_new)

		except:
			print('Only one step steering happened')

	def UV_go_diff(self,x,y,uv=False,LED=0):
		p1,p2,q1,q2,l = self.get_pins(LED_id = LED)
		d1,d2=x,y

		s = 8 # This pin Puts Motors in Sleep or active
		GPIO.output(s, True)
		# turn off the light before any movement
		GPIO.output(l,False)

		if d1!=0 or d2 !=0:
			
			self.motor(d1,p1,p2)
			self.motor(d2,q1,q2)	

		if uv:
			GPIO.output(l,True)
		else:
			GPIO.output(l,False)

		GPIO.output(s, False)

	def one_step_diff_steer(self, LED=0, X=0,Y=0): # one step steer with IR positioning
		#print('^^^ One Step starting: x,y', x,y)

		self.UV_LED_off( LED)
		self.report_light_status(LED, 'off',X,Y)
		time.sleep(wait_before_photo)
		#print('Step 1: IR is off. Now can take frame for background')
		self.report_light_status(LED, 'transition',X,Y)
		# Step 1: Move to the new location and then turn the light on
		#self.UV_go(LED,x,y,light_on=False)
		self.UV_go_diff(X,Y,uv=False,LED=LED)
		
		#time.sleep(0.5)
		#self.report_light_status(LED, 'transition')
		time.sleep(wait_before_photo)
		self.UV_LED_on(LED)
		#print('waiting for sec...')
		time.sleep(wait_before_photo) # to be removed in the product code
		self.report_light_status(LED, 'on',X,Y)
		time.sleep(wait_before_photo)


	def find_loc(self,LED): 
		# reads the loc of light and update its current location
		
		# turn off all UVs
		self.UV_all_off() 

		#read top loc
		self.one_step_diff_steer(LED=LED, X=0,Y=0)
		try:
			[x,y] = self.read_IR_locations()[0]
			print('read_IR_locations, x,y', x,y)

			# Update the current position of LEDs
			self.update_light_locations(x,y,LED)
			# light_off
			self.UV_LED_off(LED)
			self.report_light_status(LED, 'off',x,y)
			print('initial location found')
			return x,y
		except:
			print('error')
			return None,None

	def read_loc(self,LED):
		# only reads the loc of light WO updating 
		
		# turn off all UVs
		self.UV_all_off() 

		#read top loc
		self.one_step_diff_steer(LED=LED, X=0,Y=0)
		try:
			[x,y] = self.read_IR_locations()[0]
			print('read_IR_locations, x,y', x,y)

			# light_off
			self.UV_LED_off(LED)
			self.report_light_status(LED, 'off',x,y)
			print('initial location found')
			return x,y
		except:
			return None,None
			print('error')

	def read_recent_loc(self,LED):
		# only reads the recent loc of light WO updating [without light flashing]
		try:
			[x,y] = self.read_IR_locations()[0]
			print('read_IR_locations, x,y', x,y)

			return x,y
		except:
			return None,None
			print('error')

	def reset_loc(self,LED):
		x,y = self.find_loc(LED)
		self.UV_go(LED,0,0,light_on=True)
		time.sleep(1)
		self.UV_go(LED,0,0,light_on=False)





if __name__ =='__main__':

	# initialize:
	setup_pins()