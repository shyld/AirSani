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

		self.setup_pins()

		# initilize variables
		self.rho0,self.phi0,self.light_on=[False, False, False, False],np.zeros(4), np.zeros(4)


	def convert_xy2degree(self,x,y,LED): # -width/2 < x < width/2
		self.rho0, self.phi0

		#x1,y1=rotate(x,y)
		# the rotation is so that the safely rotating area is towards positive x (negative x is prohibited)
		print(WaitTime)
		print(self.rho0,self.phi0)

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

		delta_rho = rho - self.rho0[LED]
		delta_phi = phi_t - self.phi0[LED]

		self.rho0[LED] = rho
		self.phi0[LED] = phi_t
		# apply limits
		d1 = -1*int(pixel_steps * delta_rho)
		d2 = int(degree_steps * delta_phi)

		print('d1,d2: ', d1,d2)


		return d1, d2, rho, phi_t


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

		s = 8 # Puts Motors in Sleep
		if uv:
			GPIO.output(l,True)
		else:
			GPIO.output(l,False)

		if x!=0 or y !=0:
			GPIO.output(s, True)
			self.motor(x,p1,p2)
			self.motor(y,q1,q2)
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


	def UV_go(self,LED,x,y,light_on):
		x = int(x)
		y = int(y)

		p1,p2,q1,q2,l = self.get_pins(LED_id = LED)
		self.UV_driver(x1= x, y1=y, uv=light_on,p1=p1,p2=p2,q1=q1,q2=q2,l=l,LED=LED)




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
