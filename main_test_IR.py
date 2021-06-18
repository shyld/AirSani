import os
import datetime
from control_codes.IR_cam_module import MyVideoCapture
import pandas as pd
import time
import pickle

path = os.path.dirname(os.path.abspath(__file__))
#print('path: ', path)
#print('shared_variables.TEST', shared_variables.TEST)
print(path+"/control_codes/process_detections.py")

#with open(path+"/control_codes/shared_csv_files/F2_log.csv","wb") as out, open(path+"/control_codes/shared_csv_files/F2_log_err.csv","wb") as err:
#subprocess.Popen(["python3 "+ path+"/control_codes/process_detections.py"],close_fds=True, shell=True) # stdout=out, stderr=err, 


#out, err = p.communicate()  # This will get you output
#print('err', out, err)

vid = MyVideoCapture(sensor_id=1)

import numpy as np
import cv2
import time
import datetime 
#import FaceDetection
#import torch
#import torchvision

width = 640#3280
height = 480#2464
rate = 5


#model1 = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
#model1 = model1.eval().cuda()
#torch.save(model1, 'model1.pth')

if True:
    print('cap open')

    while True:
        #ret_val, img = cap.read()
        ret_val, img = vid.get_processed_IR_frame()
        
        #img = np.array(img)
        #img = FaceDetection.detector(img, model1)

        cv2.imshow('Title',img)
        #writer.write(img)

        if cv2.waitKey(40) == 27:
            break

#writer.release()
vid.cap.release()
print('video saved')
print(vid.ALL_CENTER_list)

with open(path+"/control_codes/ALL_CENTERS.txt", "wb") as fp:   #Pickling
    pickle.dump(vid.ALL_CENTER_list, fp)
