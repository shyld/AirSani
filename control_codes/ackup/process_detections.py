# change the coordinate origin to the center
# map the detection boxes to the UV buckets
# Assign score


import math
import pandas as pd
import numpy as np
import datetime 
import shared_variables

shared_variables.init()
#df = pd.read_csv('arguments.csv')
#F_running = (df[df['arguments']=='F2']['value'].iloc[0]=='TRUE') & (df[df['arguments']=='F0']['value'].iloc[0]=='TRUE')
#Cam_width = int(df[df['arguments']=='Cam_width']['value'].iloc[0])
#Cam_height= int(df[df['arguments']=='Cam_height']['value'].iloc[0])
#f = int(df[df['arguments']=='Freq_F1_to_csv']['value'].iloc[0])
#Check_stop_period = int(df[df['arguments']=='Check_stop']['value'].iloc[0])
#max_file_size = int(df[df['arguments']=='csv_max_file_size']['value'].iloc[0])



# df for writing the results: initialization

#df_empty = pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})
#df_priority = df_empty
#df_priority.to_csv('unsorted_scores.csv',index=False)
a0,a1, b0, b1, c0,c1 = 0,0,0,0,0,0
#parsing_time = datetime.datetime(2010, 1, 1, 1, 1)

# update check error file
t = datetime.datetime.now()
#df = pd.DataFrame({'F1':[t], 'F2':[t], 'F3':[t]})
#df.to_csv('check_error.csv',index=False)

# A function to check the stopping rule
def check_stop(b0,b1):
	F_running = True
	if b1 != b0:
		print('Check_stop')
		b0 = b1
		df0 = pd.read_csv('config.csv')
		F_running = (df0[df0['arguments']=='F2']['value'].iloc[0]=='TRUE') & (df0[df0['arguments']=='F0']['value'].iloc[0]=='TRUE')
	return b0, F_running

# process frames until user exits
#while True:
for i in range(100):

	t = datetime.datetime.now()
	a1 =  int((t.second+t.microsecond/1e6)*shared_variables.freq_process)
	#b1 =  int(t.second/Check_stop_period)

	# Check_stop every x sec
	#b0, F_running = check_stop(b0,b1)

	# updates the priority table every x sec
	df = shared_variables.detected_coordinates

	if a1 != a0 and len(df)>0:
		a0 = a1

		for i in range(len(df)):
			#print('len(df) ' , len(df_remaining))
			x1 = math.floor((df.Left.iloc[i] - Cam_width/2)/Coverage_size)* Coverage_size
			x2 = math.floor((df.Right.iloc[i] - Cam_width/2)/Coverage_size)* Coverage_size
			y1 = math.floor((df.Top.iloc[i] - Cam_height/2)/Coverage_size)* Coverage_size
			y2 = math.floor((df.Bottom.iloc[i] - Cam_height/2)/Coverage_size)* Coverage_size

			t_detection = df.time.iloc[i]
			priority = df.priority.iloc[i]

			for i1 in np.arange(x1,x2):
				for j1 in np.arange(y1,y2):

					idx  = (shared_variables.scored_spots['i']==i1) & (shared_variables.scored_spots['j']==j1)
					df_search = shared_variables.scored_spots[idx]
					
					if len(df_search)>0:
						# update score
						new_score = np.min([df_search['score'].values[0]+1, max_score])
						shared_variables.scored_spots.loc[idx, 'score'] = new_score
						# update time
						shared_variables.scored_spots.loc[idx, 'time'] = t

					else: # if new area
						shared_variables.scored_spots.loc[idx,'score'] = min_score
						df_temp = pd.DataFrame({'time':[t_detection], 'priority':[priority], 'i':[i1], 'j':[j1],'score':[min_score]})
						shared_variables.scored_spots =  pd.concat([shared_variables.scored_spots, df_temp])

		# remove the captured areas
		idx  = (shared_variables.detected_coordinates['time']>t)
		shared_variables.detected_coordinates  = shared_variables.detected_coordinates[idx].reset_index(drop=True)
	
	## update check_error file
	#df_check = pd.read_csv('check_error.csv')
	#df_check['F2'] = pd.to_datetime(df_check['F2'])
	#df_check['F2'] = [t]
	#df_check.to_csv('check_error.csv',index=False)




