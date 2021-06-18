import pandas as pd
import cv2
import numpy as np
import time
import datetime
#from control_codes.person_detection.person_detection import person_detection
import os

#from control_codes import shared_variables
from sklearn.neighbors import NearestNeighbors

from control_codes import shared_variables
from control_codes.device.steer import UV


PATH = os.path.dirname(os.path.abspath(__file__))

my_UV = UV()


def check_human_exposure():
# check human exposure after UV on.: 
# Find UV locations, 
# check with the detected_coordinates, if they are close then turn off the UV.

	S = shared_variables.scored_spots.reset_index()
	df_UV = S[S['score']==-1]

	D_temp = shared_variables.detected_coordinates
	
	if len(df_UV)==0 or len(D_temp)==0:
		return 0

	# Convert to numpy
	D_temp['x'] = (D_temp['Left']+D_temp['Right'])/2 #- shared_variables.Cam_width/2
	D_temp['y'] = (D_temp['Top']+D_temp['Bottom'])/2 #- shared_variables.Cam_height/2
	D = D_temp[['x','y']].to_numpy()

	np_UV = df_UV[['i','j']].to_numpy()

	#print('shapes in UV_assignment: ')
	#print(D.shape)
	#print(np_UV.shape)

	#D = D[['']].
	neigh = NearestNeighbors(n_neighbors=1)
	neigh.fit(D)
	    
	D,I = neigh.kneighbors(np_UV, n_neighbors=1, return_distance=True)
	idx = np.where(D[:,0]<1*shared_variables.Coverage_size)[0]
	#print('check_human_exposure(): len(idx)', idx.shape)
	#for i in range(idx):
	if idx.shape[0]==0:
		return 0

	try:
		shared_variables.update_scores_from_file_immediate()
		#print('1 ',shared_variables.scored_spots.loc[df_UV['index'].to_numpy()[idx],'score'])
		shared_variables.scored_spots.loc[df_UV['index'].to_numpy()[idx],'score']=1
		#print('2 ', shared_variables.scored_spots.loc[df_UV['index'].to_numpy()[idx],'score'])
		shared_variables.write_scores_to_file_immediate()

		# To Modify Later
		my_UV.UV_all_off()
	except:
		print('no close points')


def check_human_exposure_2(x1,x2,y1,y2):
# check human exposure after UV on.: 
# Find UV locations, 
# check with the detected human boxes (x1,x2,y1,y2), if they are close then turn off the UV.

	#x1 = x1 - shared_variables.Cam_width/2
	#x2 = x2 - shared_variables.Cam_width/2
	#y1 = y1 - shared_variables.Cam_height/2
	#y2 = y2 - shared_variables.Cam_height/2


	S = shared_variables.scored_spots.reset_index()
	df_UV = S[S['score']==-1]

	#print('In check_human_exposure_2:', x1,x2,y1,y2)
	#print('df_UV', df_UV)
	

	if len(df_UV)==0:
		return 0

	np_UV = df_UV[['i','j']].to_numpy()
	#print('np_UV: ', np_UV)

	for i in range(np_UV.shape[0]):
		x_uv,y_uv = np_UV[i,0],np_UV[i,1]

		m = shared_variables.UV_margin

		if x1-m<x_uv and x_uv<x2+m and y1-m<y_uv and y_uv<y2+m:
			#print('TRUEEEEEEEE Exposure in SQUARE')

			shared_variables.update_scores_from_file_immediate()
			#print(df_UV.columns)
			#print(df_UV['index'])
			#print("df_UV['index'].to_numpy() ", df_UV['index'].to_numpy())
			#print('1_2: ',shared_variables.scored_spots.loc[df_UV['index'].to_numpy()[i],'score'])
			shared_variables.scored_spots.loc[df_UV['index'].to_numpy()[i],'score']=1
			#print('2_2: ',shared_variables.scored_spots.loc[df_UV['index'].to_numpy()[i],'score'])
			shared_variables.write_scores_to_file_immediate()
			
			# To Modify Later
			my_UV.UV_all_off()


def check_human_exposure_3(x,y):
# check human exposure BEFORE UV on.: 
# Input x,y as the target UV location, 
# check with the detected_coordinates, if they are close then turn off the UV.

	D_temp = shared_variables.detected_coordinates
	print('check_human_exposure_3: D_temp', D_temp)
	if len(D_temp)==0:
		return False

	# Convert to numpy
	D_temp['x'] = (D_temp['Left']+D_temp['Right'])/2 #- shared_variables.Cam_width/2
	D_temp['y'] = (D_temp['Top']+D_temp['Bottom'])/2 #- shared_variables.Cam_height/2
	D = D_temp[['x','y']].to_numpy()


	#print('shapes in UV_assignment: ')
	#print(D.shape)
	#print(np_UV.shape)

	#D = D[['']].
	neigh = NearestNeighbors(n_neighbors=1)
	neigh.fit(D)
	    
	D,I = neigh.kneighbors([[x,y]], n_neighbors=1, return_distance=True)
	#idx = np.where(D[0,0]<2*shared_variables.Coverage_size)[0]

	if D[0,0]<1*shared_variables.Coverage_size: 
		return True
	else: 
		return False


def check_human_exposure_4(x,y):
# check human exposure BEFORE UV on.: 
# Input x,y as the target UV location, 
# check with the detected_coordinates, if they are close then turn off the UV.
	for box in shared_variables.moving_people:
		x1,x2 = min(box[0],box[2]), max(box[0],box[2])
		y1,y2 = min(box[1],box[3]), max(box[1],box[3])

		#x1 = x1 - shared_variables.Cam_width/2
		#x2 = x2 - shared_variables.Cam_width/2
		#y1 = y1 - shared_variables.Cam_height/2
		#y2 = y2 - shared_variables.Cam_height/2

		m = shared_variables.UV_margin
		if x1-m<x and x<x2+m and y1-m<y and y<y2+m:
			#print('XXXXXXX      XXXXXXX')
			return True
		#exposure = True
	return False

