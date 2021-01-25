import pandas as pd
import cv2
import numpy as np
import time
import datetime
#from control_codes.person_detection.person_detection import person_detection
import os


from sklearn.neighbors import NearestNeighbors

from control_codes import shared_variables

PATH = os.path.dirname(os.path.abspath(__file__))


def check_human_exposure():

	S = shared_variables.scored_spots
	df_UV = S[S['score']==-1]

	D = shared_variables.detected_coordinates

	if len(df_UV)==0 or len(D)==0:
		return 0

	#D = D[['']].
	neigh = NearestNeighbors(n_neighbors=2)
	neigh.fit(D)
	    
	D,I = neigh.kneighbors(df_UV, n_neighbors=2, return_distance=True)
	idx = np.where(D[:,0]<2*shared_variables.Coverage_size)[0]
	print('check_human_exposure(): len(idx)', idx.shape)
	shared_variables.scored_spots.loc[idx,'score']=1
