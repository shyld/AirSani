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

WaitTime = .0005
theta=[0,0,0,0]
degree_steps = 64/5.62/2 *1.2
pixel_steps = (64/5.62/2)/13.8*5.5*0.8
wait_before_photo = 2 # sec

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
		self.phi0= np.load(path+'UV_coordinates_phi.npy')
		self.rho0= np.load(path+'UV_coordinates_rho.npy')
		#self.x0 = 0
		#self.y0 = 0
		
		self.light_on =[False, False, False, False]

		# Initialize the locations

		# A function to initialize only one LED, keeping the other LEDs off



	def convert_xy2degree(self,x,y,LED): # -width/2 < x < width/2
		print('convert_xy2degree', x,y,)
		#x1,y1=rotate(x,y)
		# the rotation is so that the safely rotating area is towards positive x (negative x is prohibited)
		#print(WaitTime)
		#print(self.rho0,self.phi0)

		rho = np.sqrt(x**2+y**2)
		phi = np.arctan2(y,x)/np.pi*180
		#print('rho,phi: ', rho,phi)
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
		

		delta_rho = rho - self.rho0[LED]
		delta_phi = phi_t - self.phi0[LED]
		print('rho , self.rho0[LED], delta_rho', int(rho) , int(self.rho0[LED]), int(delta_rho))
		print('phi_t, self.phi0[LED], delta_rho', int(phi_t), int(self.phi0[LED]), int(delta_phi))

		self.rho0[LED] = rho
		self.phi0[LED] = phi_t

		#np.save(path+'UV_coordinates_phi', self.phi0)
		#np.save(path+'UV_coordinates_rho', self.rho0)

		# adjust steps based on polar param
		d1 = -1*int(pixel_steps * delta_rho) #(-1 for direction)
		d2 = int(degree_steps * delta_phi)

		#print('d1,d2: ', d1,d2)
		#print('delta_phi: ', delta_phi)
		#print('self.phi0[LED]: ', self.phi0[LED])

		return d1, d2, rho, phi_t

	def update_light_locations(self,x,y,LED):
		d1, d2, rho, phi_t = self.convert_xy2degree(x,y,LED)
		#print()
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

	def UV_go_polar(self,rho,phi,uv=False,LED=0):
		p1,p2,q1,q2,l = self.get_pins(LED_id = LED)
		d1,d2=rho,phi

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
			q1,q2 = 18, 23
			l = 9  
		return p1,p2,q1,q2,l


	def UV_go(self,LED,x,y,light_on): # LED \in {0,1,2,3}
		x = int(x)
		y = int(y)

		p1,p2,q1,q2,l = self.get_pins(LED_id = LED)
		self.UV_driver(x1= x, y1=y, uv=light_on,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=LED)
		print('UV_go',LED,x,y,light_on)


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
			print('find_nearest_loc, x,y,centers ', x,y, centers )
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centers)
			distances, indices = nbrs.kneighbors([[x,y]])
			#print(distances)
			# if the light is close
			neighborhood = 150
			if distances[0,0]<neighborhood:
				print(centers)
				return centers[indices[0,0]]
			else:
				return None

	def report_light_status(self, LED, light_status):
		# write in to the CSV file
		PATH = os.path.dirname(os.path.abspath(__file__))
		try:
			df_status = pd.read_csv(PATH+'/light_status.csv')
		except:

			df_status = pd.DataFrame({'LED0': [False],'LED1': [False],'LED2': [False],'LED3': [False]})

		df_status['LED'+str(LED)] = [light_status]
		#df_check[s] = [datetime.datetime.now()]
		df_status.to_csv(PATH+'/light_status.csv',index=False)


	def one_step_steer(self, LED, x,y): # one step steer with IR positioning
		#print('^^^ One Step starting: x,y', x,y)
		x = int(x)
		y = int(y)

		self.UV_LED_off( LED)
		self.report_light_status(LED, 'off')
		time.sleep(wait_before_photo)
		#print('Step 1: IR is off. Now can take frame for background')
		# Step 1: Move to the new location and then turn the light on
		self.report_light_status(LED, 'transition')
		self.UV_go(LED,x,y,light_on=False)
		#time.sleep(3)
		
		time.sleep(wait_before_photo)
		self.UV_LED_on(LED)
		#print('LED on')
		time.sleep(wait_before_photo) # to be removed in the product code
		self.report_light_status(LED, 'on')
		time.sleep(wait_before_photo)
		#time.sleep(0.5)
		# Update the current position of LEDs (x,y) in the internal class variables
		self.update_light_locations(x,y,LED)
		#print('report_light_status(LED, True)')
		#print('self.rho0[LED],self.phi0[LED] ', self.rho0[LED],self.phi0[LED])
		#print('*** one step ended ***')


	def two_step_steer(self, LED, x,y): # the light will automatically turn on
		
		self.one_step_steer(LED, x,y)
		print('One Step Steering Done')
		# check if the new light location is far from the center, otherwise don't need two-step steering
		if (x**2 + y**2)<20**2:
			print('light in the center area')
			return 0
		#print('waiting for 1 sec before taking step 2')
		time.sleep(wait_before_photo)
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


	def one_step_polar_steer(self, LED=0, rho=0,phi=0): # one step steer with IR positioning
		#print('^^^ One Step starting: x,y', x,y)

		self.UV_LED_off( LED)
		self.report_light_status(LED, 'off')
		time.sleep(wait_before_photo)
		print('Step 1: IR is off. Now can take frame for background')
		self.report_light_status(LED, 'transition')
		# Step 1: Move to the new location and then turn the light on
		#self.UV_go(LED,x,y,light_on=False)
		self.UV_go_polar(rho,phi,uv=False,LED=LED)
		
		#time.sleep(0.5)
		#self.report_light_status(LED, 'transition')
		time.sleep(wait_before_photo)
		self.UV_LED_on(LED)
		print('waiting for sec...')
		time.sleep(wait_before_photo) # to be removed in the product code
		self.report_light_status(LED, 'on')
		time.sleep(wait_before_photo)
		#time.sleep(0.5)
		# Update the current position of LEDs (x,y) in the internal class variables
		#self.update_light_locations(x,y,LED)
		print('report_light_status(LED, True)')
		#print('self.rho0[LED],self.phi0[LED] ', self.rho0[LED],self.phi0[LED])
		print('*** one step ended ***')

		# return only one location


	def init_loc(self,LED):
		
		# turn off all UVs
		self.UV_all_off() 

		#read top loc
		self.one_step_polar_steer(LED=LED, rho=0,phi=0)
		try:
			[x,y] = self.read_IR_locations()[0]
			print('read_IR_locations, x,y', x,y)
			if (x**2 + y**2)<100**2:
				location_found = True
				#self.update_light_locations(x,y,LED)
				#if self.rho0[LED]<=0:
				#	s = 1
				#else:
				#	s = -1
			else:
				location_found = False
		except: 
			location_found = False

		if location_found == False:
			print('location_unknown')
			s = 1
			for j in range(2):
				print('j', j)
				i=0
				s = s* -1
				x = 100
				y = 100
				while (x**2 + y**2)>100**2 and i<10:
					i+=1
					print('searching step: ',i)
					print('light in the center area')
				# reducce rho
					print('self.rho0[LED]',self.rho0[LED])
					self.one_step_polar_steer(LED=LED, rho=s*50,phi=0)

				# recheck the loc of IR
					try:
						[x,y] = self.read_IR_locations()[0]
					except:
						print('init_loc: spot not found')
						x = 100
						y = 100
				
				if i<9:
					location_found = True
					break

		if location_found == True:
			#self.update_light_locations(x,y,LED)
			#	if self.rho0[LED]<=0:
			#		s = 1
			#	else:
			#		s = -1

			i=0
			while (x**2 + y**2)<30**2 and i<4:
				i+=1
				print('searching step: ',i)
				print('light in the center area')
			# increase rho
				self.one_step_polar_steer(LED=LED, rho=50,phi=0)

			# recheck the loc of IR
				try:
					[x,y] = self.read_IR_locations()[0]
				except:
					print('init_loc: spot not found')

			# Update the current position of LEDs
			self.update_light_locations(x,y,LED)
			# light_off
			self.UV_all_off()
			self.report_light_status(LED, 'off')
			print('initial location found')
			return x,y
		
		else:
			print('LED not initialized')
			return None

	def reset_loc(self,LED):
		
		x,y = self.init_loc(LED)
		self.UV_go(LED,0,0,light_on=False)





if __name__ =='__main__':

	# initialize:
	setup_pins()
