import pandas as pd
import datetime 
import os
import numpy as np
#global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_current_coordinates
PATH = os.path.dirname(os.path.abspath(__file__))


def init():
	######### Setting Variables ########
	global Cam_width, Cam_height, Coverage_size, min_score, max_score, UV_after_sec, freq_process, TEST, UV_margin, t_no_person
	Cam_width = 640
	Cam_height = 480
	Coverage_size = 50
	min_score = 1
	max_score = 10
	UV_after_sec = 1
	freq_process = 2
	TEST = False
	UV_margin = 50 # margin to avoid UV exposure
	t_no_person = 15



	######### Detection Variables ########
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates, moving_people
	detected_coordinates = pd.DataFrame({'time':[], 'priority':[] , 'Left':[], 'Right':[],'Top':[], 'Bottom':[]})
	UV_wall = pd.DataFrame({'time':[], 'x1':[], 'y1':[],'x2':[], 'y2':[]})
	avoid_list = pd.DataFrame({'time':[], 'Left':[], 'Right':[],'Top':[], 'Bottom':[]})
	scored_spots = pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})
	UV_coordinates = [[0,0],[0,0],[0,0],[0,0]]
	moving_people = np.array([])

	# internal variables
	global tf_detection, tf_scores, tf_UV
	tf_detection=0
	tf_scores=0
	tf_UV = 0

	global UV_spots
	UV_spots = [0,0,0,0]


def add_detections(boxes, priority):
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	t = datetime.datetime.now()
	for i in range(boxes.shape[0]):
		#L=[]
		#L.append(boxes)
		df_single_box = pd.DataFrame({'time':[t], 'priority':[priority], 'Left':[int(boxes[i,0])], 'Right':[int(boxes[i,2])],'Top':[int(boxes[i,1])], 'Bottom':[int(boxes[i,3])]})
		detected_coordinates = pd.concat([ df_single_box, detected_coordinates])
	
def add_people(moving_people_boxes):
	global moving_people
	moving_people = moving_people_boxes





def add_avoid_list(boxes):
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	t = datetime.datetime.now()
	# add the new points
	for i in range(boxes.shape[0]):
		df_single_box = pd.DataFrame({'time':[t], 'Left':[int(boxes[i,0])], 'Right':[int(boxes[i,2])],'Top':[int(boxes[i,1])], 'Bottom':[int(boxes[i,3])]})
		avoid_list = pd.concat([avoid_list, df_single_box])

def remove_old_avoid_list():
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	t = datetime.datetime.now()
	avoid_list = avoid_list[avoid_list['time']>=t-datetime.timedelta(seconds=4)]
	#print('remove')

def remove_old_detection_list():
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	t = datetime.datetime.now()
	detected_coordinates = detected_coordinates[pd.to_datetime(detected_coordinates['time'])>=t-datetime.timedelta(seconds=2)]
	#print('remove')

def remove_old_scored_list():
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	t = datetime.datetime.now()
	scored_spots = scored_spots[pd.to_datetime(scored_spots['time'])>=t-datetime.timedelta(seconds=20)]


# For Mohammad to update:
def add_still_people(still_centers):
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	t = datetime.datetime.now()
	for i in range(still_centers.shape[0]):
		print('still_centers[i]',still_centers[i])
		#df_single_box = pd.DataFrame({'time':[t], 'priority':[priority], 'Left':[int(boxes[i,0])], 'Right':[int(boxes[i,2])],'Top':[int(boxes[i,1])], 'Bottom':[int(boxes[i,3])]})
		#detected_coordinates = pd.concat([detected_coordinates, df_single_box])

def write_detections_to_file():
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates,tf_detection,tf_scores

	t = datetime.datetime.now()
	t_s =  int(t.second)
	if t_s!= tf_detection:
		tf_detection = t_s
		try:
			df = pd.read_csv(PATH+'/shared_csv_files/detected_coordinates.csv')
			#print(pd.to_datetime(df_new['time'].astype('datetime64[ns]'))<t)
			df_new =  pd.concat([df, detected_coordinates])
		except:
			df_new=detected_coordinates 
			#print(pd.to_datetime(df_new['time'].astype('datetime64[ns]'))<t)

		#print(t, t-datetime.timedelta(seconds=4), df_new['time'][0])
		#print(pd.to_datetime(df_new['time'].astype('datetime64[ns]'))<t)

		df_new = df_new[pd.to_datetime(df_new['time'])<t-datetime.timedelta(seconds=4)]

		detected_coordinates=detected_coordinates[pd.to_datetime(detected_coordinates['time'])>=t-datetime.timedelta(seconds=4)]
		#detected_coordinates = pd.DataFrame({'time':[], 'priority':[] , 'Left':[], 'Right':[],'Top':[], 'Bottom':[]})
		df_new.to_csv(PATH+'/shared_csv_files/detected_coordinates.csv',index=False)
		#df_priority = pd.read_csv('scored_spots.csv')
def update_scores_from_file():

	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates,tf_detection,tf_scores
	t = datetime.datetime.now()
	t_s =  int(t.second)
	#print('read_scores_from_file(), ', t_s, tf_scores)
	if t_s!= tf_scores:
		#print('in the if *******************')
		tf_score = t_s
		try:
			scored_spots = pd.read_csv(PATH+'/shared_csv_files/scored_spots.csv')
			#scored_spots =  pd.concat([df, scored_spots])
			#print('in try: scored_spots len', len(scored_spots))
		except:
			#print('in except: scored_spots len', len(scored_spots))
			scored_spots=pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})
		
		#df_empty = pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})
		#df_empty.to_csv(PATH+'/shared_csv_files/scored_spots.csv',index=False)

def update_scores_from_file_immediate():

	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates,tf_detection,tf_scores
	t = datetime.datetime.now()
	t_s =  int(t.second)
	#print('read_scores_from_file(), ', t_s, tf_scores)
	if True:
		#print('in the if *******************')
		tf_score = t_s
		try:
			scored_spots = pd.read_csv(PATH+'/shared_csv_files/scored_spots.csv')
			#scored_spots =  pd.concat([df, scored_spots])
			#print('in try: scored_spots len', len(scored_spots))
		except:
			#print('in except: scored_spots len', len(scored_spots))
			scored_spots=pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})

def write_scores_to_file_immediate():
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates,tf_detection,tf_scores

	t = datetime.datetime.now()
	t_s =  int(t.second)
	if True:
		tf_detection = t_s
		scored_spots.to_csv(PATH+'/shared_csv_files/scored_spots.csv',index=False)


def test_variables():
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	print('test_variables(): detected_coordinates ',detected_coordinates )
#print(detected_coordinates)