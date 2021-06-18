
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from control_codes import shared_variables
import cv2
import time
#try:
if False:
	from control_codes.touch_detection.position_estimation_class_v2 import pose_estimation
	ON_Jetson = True
else:
#except:
	print('no pose estimation for person detection')
	ON_Jetson = False
#from control_codes.drawing_functions import origin_corner_2_center
# Parameters
contour_lower = 900 
contour_upper = 500000
predefined_distance=1000
#t_no_person = 10 # seconds
still_threshold = 0.05
t_still = 5 # sec
shared_variables.TEST = True
# t_still_frq = 10/5

# This class returns the boxes of detected people. The output is a n by 10 matrix, where the first
# 4 columns are the x1,y1,x2,y2 and the fifth column is the time since last movement and
# the last 5 columns are the distances to the previous locations at 5 last sec.


def origin_center_2_corner(x1,x2,y1,y2):
	xp1 = max(int(x1 + shared_variables.Cam_width/2),0)
	xp2 = min(int(x2 + shared_variables.Cam_width/2),shared_variables.Cam_width)
	yp1 = max(int(shared_variables.Cam_height/2 -y1),0)
	yp2 = min(int(shared_variables.Cam_height/2 -y2),shared_variables.Cam_height)
	return xp1,xp2,yp1,yp2

def origin_corner_2_center(x1,x2,y1,y2):
	xp1 = int(x1 - shared_variables.Cam_width/2)
	xp2 = int(x2 - shared_variables.Cam_width/2)
	yp1 = int(shared_variables.Cam_height/2-y1)
	yp2 = int(shared_variables.Cam_height/2-y2)
	return xp1,xp2,yp1,yp2



class person_detection:

	def __init__(self, person_scale=0.1):
		self.person_scale = person_scale
		self.persons_previous = np.array([])
		self.current_time  = 0
		#try:
		if False:
			self.my_pose_estimation = pose_estimation()
		#except:
			print('no pose estimation initialized')

	def find_moving_people(self, L=[]): 
		# L is the list of rectangles of moving contours
		# person_scale is the average shoulder to waist length of the people
		if len(L)==0:
			#print('find_moving_people: L:[]')
			return np.array([])
		# Find the contour rectangle centers
		X_centers = [(x1+x2)/2 for (x1,y1,x2,y2) in L]
		Y_centers = [(y1+y2)/2 for (x1,y1,x2,y2) in L]
		weights = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in L]

		centers = np.zeros((len(X_centers),2)) # a np array of centers
		centers[:,0], centers[:,1] = np.array(X_centers), np.array(Y_centers)

		# Apply DBSCAN clustering
		try:
			#print('np.array(weights).shape ', np.array(weights).shape)
			db = DBSCAN(eps=0.7*self.person_scale, min_samples=1).fit(centers, y=None, sample_weight = np.array(weights))
			clusters = db.labels_

			# find centers of the clusters
			cluster_centers = [[np.mean(centers[clusters==i,0]),np.mean(centers[clusters==i,1])] for i in range(np.max(clusters))]
			#print('cluster_centers: ', cluster_centers)
			# find human boxes: x0,y0,x1,y1
			C = cluster_centers
			p = self.person_scale*1.2
			# Box coordinates: x1,y1,x2,y2, time, previous distances
			Boxes = [[int(C[i][0]-p), int(C[i][1]-p), int(C[i][0]+p),int(C[i][1]+p), int(time.time())%1000,int(predefined_distance),0,0,0,0,0,0,0,0] for i in range(len(C))] 
		except:
			#print('find_moving_people: EXCEPT')
			Boxes = []
		#return centers, clusters
		return np.array(Boxes)


	def update_all_people(self, moving_persons_boxes, frame):


		final_list = []
		NEW = moving_persons_boxes
		OLD = self.persons_previous
		#print('self.persons_previous', self.persons_previous)

		# Special Cases
		if OLD.shape[0]==0:
			self.persons_previous = NEW
			return NEW

		if NEW.shape[0]>0:
			# Apply nearest neghbor search on the upper left coordinate
			nbrs = NearestNeighbors(n_neighbors=1).fit(NEW[:,:2])
			distances, indices = nbrs.kneighbors(OLD[:,:2])
			I_common_OLD = np.where(distances<shared_variables.Coverage_size)[0] # index of the moving items
			#print('person_detection: NEW.tolist()', NEW.tolist()[0])
			
			for i in range(NEW.shape[0]):
				final_list.append(NEW[i,:].tolist())

		else:
			I_common_OLD = np.array([])


			#return OLD
		
		
		t_now = int(time.time())%1000
		
		# Update time
		#final_list = np.array(final_list)
		#print('final_list.shape: ',np.array(final_list).shape)
		for i in range(len(final_list)):
			final_list[i][4] = t_now
		#[for final_list[:,4] = t_now]
		#final_list = final_list.tolist()
		

		#print('person_detection: t_now',t_now)
		#print('person_detection:,OLD[0,:]', OLD[0,:])
		#self.current_time = t_now


		I_no_person = np.where((t_now-OLD[:,4])%1000>shared_variables.t_no_person)[0] #if the box contains no person (the last update time is long time ago)
		
		# check if there is actually a still person inside the box
		for idx in I_no_person:
			x1,y1,x2,y2 = OLD[idx][0],OLD[idx][1],OLD[idx][2],OLD[idx][3]
			x1,y1,x2,y2 = origin_center_2_corner(x1,x2,y1,y2)
			x1,y1,x2,y2 = min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)
			
			# Run poes estimation
			if ON_Jetson:
				#print(x1,x2,y1,y2)
				#print(frame.shape)
				#print(frame[y2:y1].shape)
				#print(frame[y1:y2,x1:x2].shape)
				#print('frame[y1:y2,x1:x2,:].shape: ', frame[y1:y2,x1:x2,:].shape)
				(person_count , L_all, L_touch, frame) = self.my_pose_estimation.get_keypoints(frame[y1:y2,x1:x2,:])
			else:
				person_count = 0

			#except:
			#	person_count = 0
			
			# remove those containing person
			if person_count>0:
				I_no_person.remove(idx)
				# reset its time
				OLD[idx,4] = t_now


		#print('I_common_OLD ',I_common_OLD)
		#print('time from OLD',(t_now-OLD[:,4])%1000)
		#print('I_no_person ', I_no_person)
        
        # I_complement_OLD are those boxes from the previous frame, infered to contain still people
		I_complement_OLD = [i for i in range(len(OLD)) if (not (i in I_common_OLD)) and (not (i in I_no_person))]

		#print('I_complement_OLD: ', I_complement_OLD)
		
		# Merge the OLD and NEW Boxes to the final_list
		T = OLD[I_complement_OLD,:].tolist()
		for i in range(len(T)):
			final_list.append(T[i])
		#if len(I_complement_OLD)>0:
			
		
		final_list = np.array(final_list)

		# Update the distances from the previous positions as well as time for the moving boxes
		nbrs = NearestNeighbors(n_neighbors=1).fit(OLD[:,:2])
		if final_list.ndim==1:
			final_list= np.reshape(final_list, (1,2))
			
		distances, indices = nbrs.kneighbors(final_list[:,:2])
		
		# update the recent movement distances every second
		if (t_now >self.current_time):
			self.current_time = t_now
			final_list[:,5:8] = final_list[:,6:9]

		final_list[:,-1] = np.reshape(distances,(-1,))
			# update time
		#final_list[:,4] = t_now
		
		# Update previous persons list
		self.persons_previous = final_list

		return final_list

	# Find the positions of still (possibly sitting people)
	def find_still_people(self, all_people=[]):
		if all_people.shape[0]==0:
			return(np.array([]))
		#print('in find_still_people: ')
		#print(all_people[:,5:9].shape)
		#print(np.where(np.mean(all_people[:,5:9],axis=1)<still_threshold)[0])
		#print(all_people.shape)
		T_temp = all_people[np.where(np.mean(all_people[:,5:9],axis=1)<still_threshold)[0],:4]
		return np.concatenate((np.reshape((T_temp[:,0]+T_temp[:,2])/2,(-1,1)), np.reshape((T_temp[:,1]+T_temp[:,3])/2,(-1,1))),axis=1)
	

	def shadow_remove(self,img):
		rgb_planes = cv2.split(img)
		result_norm_planes = []
		for plane in rgb_planes:
			dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
			bg_img = cv2.medianBlur(dilated_img, 21)
			diff_img = 255 - cv2.absdiff(plane, bg_img)
			norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			result_norm_planes.append(norm_img)
		shadowremov = cv2.merge(result_norm_planes)
		return shadowremov

	def change_coordinate(self,X):
		try:
			X[:,0] = np.floor(X[:,0]-shared_variables.Cam_width/2)
			X[:,1] = np.floor(X[:,1]-shared_variables.Cam_height/2)
			X[:,2] = np.floor(X[:,2]-shared_variables.Cam_width/2)
			X[:,3] = np.floor(X[:,3]-shared_variables.Cam_height/2)
		except:
			print('no input in change coordinate')
		return X


	# the main function
	def get_all_people(self, frame1, frame2):
		#print('In person_detection, get_all ... :frame1.shape', frame1.shape)
		#frame1 = self.shadow_remove(frame1)
		#frame2 = self.shadow_remove(frame2)

		# find motion contours
		diff = cv2.absdiff(frame1, frame2)
		gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5,5), 0)
		_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
		dilated = cv2.dilate(thresh, None, iterations=3)
		contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		c1 = frame1.shape[0]
		contour_lower = c1*c1/3500
		contour_upper = c1*c1/ 10

		# Find motion boxes
		L = []
		for contour in contours:
			(x, y, w, h) = cv2.boundingRect(contour)
			if cv2.contourArea(contour) < contour_lower or cv2.contourArea(contour) > contour_upper :
				continue


			x1,x2,y1,y2 = origin_corner_2_center(x,x+w,y,y+h)
			L.append((x1, y1, x2, y2))
			#print('Original x,y', x,y)
			#print('Converted x,y', x1,y1)
		#print('in person detection: L', L)
		#if len(L)>0:
		# Find moving people boxes based on motion boxes
		try:
		#if True:
			M = self.find_moving_people(L=L)
			#print('M: try', M)
		except:
		#if False:
			M = np.array([])
			#print('M: except')

		# Update people boxes
		try:
		#if True:
			A = self.update_all_people(moving_persons_boxes=M, frame = frame1)
			#print('A: try')
		except:
		#if False:
			A = np.array([])
			#print('A: except')

		#else:
		#	M = np.array([])
		#	A = np.array([])
		#print('find_moving_people: ', M)
		#print('update_all_people: ', A)

		#S = self.find_still_people(all_people=A)
		S = np.array([])

		# Change the origin to center:
		L = np.array(L)
		#L = self.change_coordinate(L)
		#A = self.change_coordinate(A)
		#S = self.change_coordinate(S)
		#M = self.change_coordinate(M)

		# return numpy arrays
		return  A, L, S, M # moving people, moving areas, centers of the still people [all in numpy array]











		
