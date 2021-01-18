import pandas as pd
import datetime 

#global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_current_coordinates


def init():
	######### Setting Variables ########
	global Cam_width, Cam_height, Coverage_size, min_score, max_score, UV_after_sec, freq_process
	Cam_width = 640
	Cam_heigh = 480
	Coverage_size = 30
	min_score = 1
	max_score = 10
	UV_after_sec = 1
	freq_process = 2



	######### Detection Variables ########
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	detected_coordinates = pd.DataFrame({'time':[], 'priority':[] , 'Left':[], 'Right':[],'Top':[], 'Bottom':[]})
	UV_wall = pd.DataFrame({'time':[], 'x1':[], 'y1':[],'x2':[], 'y2':[]})
	avoid_list = pd.DataFrame({'time':[], 'Left':[], 'Right':[],'Top':[], 'Bottom':[]})
	scored_spots = pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})
	UV_coordinates = [[0,0],[0,0],[0,0],[0,0]]

def add_detections(boxes, priority):
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	t = datetime.datetime.now()
	for i in range(boxes.shape[0]):
		#L=[]
		#L.append(boxes)
		df_single_box = pd.DataFrame({'time':[t], 'priority':[priority], 'Left':[int(boxes[i,0])], 'Right':[int(boxes[i,2])],'Top':[int(boxes[i,1])], 'Bottom':[int(boxes[i,3])]})
		detected_coordinates = pd.concat([ df_single_box, detected_coordinates])
	
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
	detected_coordinates = detected_coordinates[detected_coordinates['time']>=t-datetime.timedelta(seconds=2)]
	#print('remove')

def remove_old_scored_list():
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	t = datetime.datetime.now()
	scored_spots = scored_spots[scored_spots['time']>=t-datetime.timedelta(seconds=5)]


# For Mohammad to update:
def add_still_people(still_centers):
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	t = datetime.datetime.now()
	for i in range(still_centers.shape[0]):
		print('still_centers[i]',still_centers[i])
		#df_single_box = pd.DataFrame({'time':[t], 'priority':[priority], 'Left':[int(boxes[i,0])], 'Right':[int(boxes[i,2])],'Top':[int(boxes[i,1])], 'Bottom':[int(boxes[i,3])]})
		#detected_coordinates = pd.concat([detected_coordinates, df_single_box])

def test_variables():
	global detected_coordinates, UV_wall, avoid_list, scored_spots, UV_coordinates
	print('test_variables(): ',detected_coordinates )
#print(detected_coordinates)