
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import cv2
import time

# Parameters
contour_lower = 900 
contour_upper = 500000
predefined_distance=1000
t_no_person = 2 # seconds
still_threshold = 0.05
t_still = 5 # sec
#t_still_frq = 10/5

# This class returns the boxes of detected people. The output is a n by 10 matrix, where the first
# 4 columns are the x1,y1,x2,y2 and the fifth column is the time since last movement and
# the last 5 columns are the distances to the previous locations at 5 last sec.

class person_detection:

	def __init__(self, person_scale=0.1):
		self.person_scale = person_scale
		self.persons_previous = np.array([])
		self.current_time  = 0

	def find_moving_people(self, L=[]): 
		# L is the list of rectangles of moving contours
		# person_scale is the average shoulder to waist length of the people
		if len(L)==0:
			print('find_moving_people: L:[]')
			return np.array([])
		# Find the contour rectangle centers
		X_centers = [(x1+x2)/2 for (x1,y1,x2,y2) in L]
		Y_centers = [(y1+y2)/2 for (x1,y1,x2,y2) in L]
		weights = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in L]

		centers = np.zeros((len(X_centers),2)) # a np array of centers
		centers[:,0], centers[:,1] = np.array(X_centers), np.array(Y_centers)

		# Apply DBSCAN clustering
		try:
			print('np.array(weights).shape ', np.array(weights).shape)
			db = DBSCAN(eps=0.7*self.person_scale, min_samples=1).fit(centers, y=None, sample_weight = np.array(weights))
			clusters = db.labels_

			# find centers of the clusters
			cluster_centers = [[np.mean(centers[clusters==i,0]),np.mean(centers[clusters==i,1])] for i in range(np.max(clusters))]
			print('cluster_centers: ', cluster_centers)
			# find human boxes: x0,y0,x1,y1
			C = cluster_centers
			p = self.person_scale*1.2
			Boxes = [[int(C[i][0]-p), int(C[i][1]-p), int(C[i][0]+p),int(C[i][1]+p), int(time.time())%1000,int(predefined_distance),0,0,0,0,0,0,0,0] for i in range(len(C))] 
		except:
			print('find_moving_people: EXCEPT')
			Boxes = []
		#return centers, clusters
		return np.array(Boxes)


	def update_all_people(self, moving_persons_boxes=np.array([])):


		final_list = []
		NEW = moving_persons_boxes
		OLD = self.persons_previous
		#print('OLD', OLD)

		# Special Cases
		if OLD.shape[0]==0:
			self.persons_previous = NEW
			return NEW

		if NEW.shape[0]>0:
			# Apply nearest neghbor search on the upper left coordinate
			nbrs = NearestNeighbors(n_neighbors=1).fit(NEW[:,:2])
			distances, indices = nbrs.kneighbors(OLD[:,:2])
			I_common_OLD = np.where(distances<1.2*self.person_scale)[0] # index of the moving items
			#print('person_detection: NEW.tolist()', NEW.tolist()[0])
			
			for i in range(NEW.shape[0]):
				final_list.append(NEW[i,:].tolist())

		else:
			I_common_OLD = np.array([])


			#return OLD
		
		
		t_now = int(time.time())%1000
		#print('person_detection: t_now',t_now)
		#print('person_detection:,OLD[0,:]', OLD[0,:])
		#self.current_time = t_now

		I_no_person = np.where((t_now-OLD[:,4])%1000>t_no_person)[0] #if the box contains no person (the last update time is long time ago)
		#print('I_common_OLD ',I_common_OLD)
		#print('I_no_person ', I_no_person)
        
        # I_complement_OLD are those boxes from the previous frame, infered to contain still people
		I_complement_OLD = [i for i in range(len(OLD)) if (not (i in I_common_OLD)) and (not (i in I_no_person))]

		# Merge the OLD and NEW Boxes to the final_list
		if len(I_complement_OLD)>0:
			final_list.append(OLD[I_complement_OLD,:].tolist()[0])
		
		final_list = np.array(final_list)

		# Update the distances from the previous positions as well as time for the moving boxes
		nbrs = NearestNeighbors(n_neighbors=1).fit(OLD[:,:2])
		distances, indices = nbrs.kneighbors(final_list[:,:2])
		
		# update the recent movement distances every second
		if (t_now >self.current_time):
			self.current_time = t_now
			final_list[:,5:8] = final_list[:,6:9]

		final_list[:,-1] = np.reshape(distances,(-1,))
			# update time
		final_list[:,4] = t_now
		
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
			L.append((x, y, x+w, y+h))
		print('in person detection: L', L)
		# Find moving people boxes based on motion boxes
		M = self.find_moving_people(L=L)

		print('in person detection: M', M)
		# Update people boxes
		A = self.update_all_people(moving_persons_boxes=M)

		S = self.find_still_people(all_people=A)

		return  A, np.array(L), S, M # moving people, moving areas, centers of the still people [all in numpy array]











		
