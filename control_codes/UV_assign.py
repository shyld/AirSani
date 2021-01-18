# change the coordinate origin to the center
# map the detection boxes to the UV buckets
# Assign score


import math
import pandas as pd
import numpy as np
import datetime 
#from control_codes import shared_variables
import shared_variables
import os
import time 


############################# PARAMTERS ###############
shared_variables.init()




#################################################




#shared_variables.init()
print('opened UV_assign.py')
PATH = os.path.dirname(os.path.abspath(__file__))
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
for i in range(1000):
	print('i ', i)
	time.sleep(0.1)

	
	# Check_stop every x sec
	#b0, F_running = check_stop(b0,b1)

	# updates the priority table every x sec
	#df = shared_variables.detected_coordinates

	try:
		df_score = pd.read_csv(PATH+'/shared_csv_files/scored_spots.csv')
		#df_new =  pd.concat([df, detected_coordinates])
	except:
		df_score = pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})

	#df_new.to_csv(PATH+'/shared_csv_files/detected_coordinates.csv',index=False)


	# covert to correct data types
	if len(df_score)>0:
		df_score['time'] = pd.to_datetime(df_score['time'])


	if len(df)>0:
		#print('*********** shared_variables.detected_coordinates: ', len(shared_variables.detected_coordinates))


		for i in range(len(df)):
			print(i, len(df) )
			#print('len(df) ' , len(df_remaining))
			x1 = math.floor((df.Left.iloc[i] - shared_variables.Cam_width/2)/shared_variables.Coverage_size)* shared_variables.Coverage_size
			x2 = math.floor((df.Right.iloc[i] - shared_variables.Cam_width/2)/shared_variables.Coverage_size)* shared_variables.Coverage_size
			y1 = math.floor((df.Top.iloc[i] - shared_variables.Cam_height/2)/shared_variables.Coverage_size)* shared_variables.Coverage_size
			y2 = math.floor((df.Bottom.iloc[i] - shared_variables.Cam_height/2)/shared_variables.Coverage_size)* shared_variables.Coverage_size

			t_detection = df.time.iloc[i]
			priority = df.priority.iloc[i]

			for i1 in np.arange(x1,x2, shared_variables.Coverage_size):
				for j1 in np.arange(y1,y2, shared_variables.Coverage_size):
					print(i1,j1)

					idx  = (df_score['i']==i1) & (df_score['j']==j1)
					df_search = df_score[idx]
					
					if len(df_search)>0:
						# update score
						new_score = np.min([df_search['score'].values[0]+1, shared_variables.max_score])
						shared_variables.scored_spots.loc[idx, 'score'] = new_score
						# update time
						shared_variables.scored_spots.loc[idx, 'time'] = t

					else: # if new area
						df_score.loc[idx,'score'] = shared_variables.min_score
						df_temp = pd.DataFrame({'time':[t_detection], 'priority':[priority], 'i':[i1], 'j':[j1],'score':[shared_variables.min_score]})
						df_score =  pd.concat([df_score, df_temp])

		# remove the captured areas 
		idx  = (df['time']>t)
		df  = df[idx].reset_index(drop=True)

		# Save updates to the file
		df.to_csv(PATH+'/shared_csv_files/detected_coordinates.csv',index=False)
		df_score.to_csv(PATH+'/shared_csv_files/scored_spots.csv',index=False)
	
	## update check_error file
	#df_check = pd.read_csv('check_error.csv')
	#df_check['F2'] = pd.to_datetime(df_check['F2'])
	#df_check['F2'] = [t]
	#df_check.to_csv('check_error.csv',index=False)




