
import numpy as np

def init_coordinates():
	phi0= np.load('UV_coordinates_phi.npy')
	rho0= np.load('UV_coordinates_rho.npy')

	print(phi0,rho0)

	rho0,phi0 = np.zeros(4), np.zeros(4)
	light_on =[False, False, False, False]
	phi0[2] = 180
	phi0[3] = 180


	np.save('UV_coordinates_phi', phi0)
	np.save('UV_coordinates_rho', rho0)

	phi0= np.load('UV_coordinates_phi.npy')
	rho0= np.load('UV_coordinates_rho.npy')

	print(phi0,rho0)