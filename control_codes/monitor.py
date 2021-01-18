# monitor
import shared_variables
import time 
shared_variables.init()

for i in range(200):
	time.sleep(1)
	print('shared_variables.detected_coordinates: ', shared_variables.detected_coordinates)
	print('shared_variables.scored_spots: ', shared_variables.scored_spots)