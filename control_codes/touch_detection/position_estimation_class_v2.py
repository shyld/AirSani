import numpy as np
import cv2
import time
import datetime 
import torch2trt

from torch2trt import TRTModule
import torch
import trt_pose

import json

import cv2
import torchvision.transforms as transforms
import PIL.Image

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

import trt_pose.models
import trt_pose.coco

import time
import os
from pathlib import Path



class pose_estimation:

	def __init__(self):
			# loading the model 
		self.person_scale = 50 # default value for person_scale

		path = os.getcwd()

		print(path)

		with open('/home/shyldai/shyld/AirSani/control_codes/touch_detection/human_pose.json', 'r') as f:
			human_pose = json.load(f)

		topology = trt_pose.coco.coco_category_to_topology(human_pose)

		num_parts = len(human_pose['keypoints'])
		num_links = len(human_pose['skeleton'])

        ## *** UNCOMMENT THE FOLLOWING IF resnet18_baseline_att_224x224_A_epoch_249_trt.pth DOES NOT EXIT ***********
		path = '/home/shyldai/shyld/model_weights/'
		model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
		MODEL_WEIGHTS = path+'resnet18_baseline_att_224x224_A_epoch_249.pth'
		model.load_state_dict(torch.load(MODEL_WEIGHTS))

		WIDTH = 224
		HEIGHT = 224
		data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


		OPTIMIZED_MODEL = path+'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

		my_file = Path(OPTIMIZED_MODEL)

		if my_file.is_file()==False:
			model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
			torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

		self.model_trt = TRTModule()
		self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
		print('The resnet model loaded')



		t0 = time.time()
		torch.cuda.current_stream().synchronize()
		for i in range(50):
			y = self.model_trt(data)
		torch.cuda.current_stream().synchronize()
		t1 = time.time()


		self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
		self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
		device = torch.device('cuda')

		
		self.parse_objects = ParseObjects(topology)
		self.draw_objects = DrawObjects(topology)


	def preprocess(self,img):
		#print('in preprocess...')
		#print(self.mean, self.std)

		global device

		WIDTH = 224
		HEIGHT = 224

		dsize=(WIDTH, HEIGHT)
		image=cv2.resize(img,dsize)

		device = torch.device('cuda')
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = PIL.Image.fromarray(image)
		image = transforms.functional.to_tensor(image).to(device)
		image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
		return image[None, ...]

####

	def plot_keypoints(self, image, Loc_all): # Loc_all[0]: x, Loc_all[1]:y, Loc_all[2]: loc_name
		width = image.shape[1]
		height= image.shape[0]
		print('Loc_all: plot_keypoints', Loc_all)
		for i in range(len(Loc_all)):
			Loc_spot = Loc_all[i]
			print('position_...py plot_keypoints: Loc_spot' ,Loc_spot)
			if Loc_spot[0]>0 and Loc_spot[1]>0:
				print('Loc_spot', Loc_spot)
				x0 = round(float(Loc_spot[1]) * width)
				y0 = round(float(Loc_spot[0]) * height)
				# Blue color in BGR
				color = (255, 0, 0)
				radius = 30
				thickness = 10
				image = cv2.circle(image, (x0, y0), radius, color, thickness)

		return image



	def get_keypoints(self,image, print_element_names = True, show_all_keypoints = True, show_list_keypoints = [10,11], show_touch_event = True, gamma = 1.0):
	    
		width = image.shape[1]
		height= image.shape[0]
	    #image = change['new']
		print('width,height',width,height)
		data = self.preprocess(image)
		cmap, paf = self.model_trt(data)
		cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
	    #print(cmap,paf)
		counts, objects,  normalized_peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)

		image_idx = 0
		person_idx = 0
		c = counts.data.cpu().numpy()[image_idx]

		OB = objects.data.cpu().numpy()[image_idx, :c , : ]
		#print(c)
		#print(objects.data.cpu().numpy()[image_idx, : , : ])

		Loc_all = []
		Loc_event = []
		image_idx = 0

		element_list = [5,6,9,10,11,12]
		element_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]
	    
		if print_element_names == True:
			print([element_names[n] for n in element_list])

		print('number of people detected: ' , c)
		if c > 0:
	    # there is an object in the first image
			for i in range(c): #for on different persons
				#print('person ', i)
				person_idx = i
				Loc_person=[]
				for j in element_list:
					body_element_type = j
					body_element_idx = objects.data.cpu().numpy()[image_idx, person_idx , body_element_type]
					element_location = normalized_peaks.data.cpu().numpy()[image_idx, body_element_type, body_element_idx, :]
					Loc_person.append(element_location.tolist())
				Loc_all.append(Loc_person) # Each row in Loc_all is a person's keypoint locations

		# Create a list of touching spots by iterating through all persons
		for i in range(len(Loc_all)):		
			Loc_person = Loc_all[i]
			# Compute Loc_event
			#c = max(np.linalg.norm(np.array(Loc_person[0])-np.array(Loc_person[1])), 
			#		np.linalg.norm(np.array(Loc_person[4])-np.array(Loc_person[5])) )
			#if c > 0:
			#	self.person_scale = c
			#dist_ref = self.person_scale
			dist_ref = 0.2

			# defualt hand location correction for undetected cases
			if np.sum(np.array(Loc_person[2]))==0:
				Loc_person[2] = [0.5, 0.5]

			if np.sum(np.array(Loc_person[3]))==0:
				Loc_person[3] = [0.5, 0.5]




			dist_tst = np.linalg.norm(np.array(Loc_person[2])-np.array([0.5, 0.5]))

			# Touch conditions
			cv2.putText(image, str(round(dist_tst,2))+' <?> ' + str(round(gamma*dist_ref,2)), (50,50+i*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

			if dist_tst > gamma*dist_ref:
				print(dist_tst,' <?>' ,gamma*dist_ref)

				
				Loc_event.append(Loc_person[2])

			dist_tst = np.linalg.norm(np.array(Loc_person[3])-np.array([0.5, 0.5]))

			cv2.putText(image, str(round(dist_tst,2))+' <?> ' + str(round(gamma*dist_ref,2)), (50,250+i*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3) #font,  fontScale, color, thickness, cv2.LINE_AA)

			if dist_tst> gamma*dist_ref:
				print(dist_tst,' <?>' ,gamma*dist_ref)
				
				Loc_event.append(Loc_person[3])


		# Plot
		print('Main Loc_all: ', Loc_all)
		final_image = image
		if len(show_list_keypoints)>0:
			if show_touch_event == False:
				print('position_estimation_class.py [show touch event FALSE]')
				print('Loc_all: ', Loc_all)
				final_image = self.plot_keypoints(image, Loc_all)
				print('final_image shape', final_image.shape)
			else:
				print('position_estimation_class.py [show touch event TRUE]')
				print('Loc_event:',Loc_event)
				final_image = self.plot_keypoints(image, Loc_event)
				print('final_image shape', final_image.shape)

		#if final_image != None:
		final_image_resized=cv2.resize(final_image,(width, height))

		return Loc_all, Loc_event, final_image_resized#cv2.cvtColor(final_image_resized, cv2.COLOR_BGR2RGB) 

	def detect_touch(self,frame):

		(L_all, L_touch, frame) = self.get_keypoints(frame)

		return frame, L_touch


