# MIT License
# Copyright (c) 2019 JetsonHacks
# See LICENSE for OpenCV license and additional information

# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV

import cv2
import numpy as np
#from csi_camera import CSI_Camera
import time
import datetime
from control_codes.find_light import find_IR
import control_codes.shared_variables
import os
try:
    from control_codes.csi_camera import CSI_Camera
    from control_codes.touch_detection.position_estimation_class_v2 import pose_estimation
except:
    print('test_mode on macOS')

import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))

ROTATE_CAM = False

show_fps = True

# preprocessing
#def pre_process(img):
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
#    blurred = cv2.GaussianBlur(img, (11, 11), 0)
#    img = cv2.threshold(blurred, 64, 255, cv2.THRESH_BINARY)[1]
#    return img

# preprocessing
def pre_process(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (width,height))
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    hist = cv2.calcHist([blurred], [0], None, [256], [0, 256]).flatten()
    target_count = 400
    summed = 0
    for i in range(255, 0, -1):
        summed += int(hist[i])
        if target_count <= summed:
            hi_thresh = i
            break
    else:
        hi_thresh = 0

    img = cv2.threshold(blurred, hi_thresh, 255, cv2.THRESH_BINARY)[1]
    
    return img

def read_light_status():
    # write in to the CSV file
    PATH = os.path.dirname(os.path.abspath(__file__)) # path of the current file (not the master file)
    try:
        #print(PATH+'/device/light_status.csv')
        df_status = pd.read_csv(PATH+'/device/light_status.csv')
        df_status = df_status[['LED0','LED1','LED2','LED3']].astype('str')
    except:

        df_status = pd.DataFrame({'LED0': [False],'LED1': [False],'LED2': [False],'LED3': [False]})

    L = df_status.iloc[0,:].tolist()
    #print('L', L)
    return L


def write_light_loc_from_IR(centers):
    PATH = os.path.dirname(os.path.abspath(__file__))    
    if len(centers[0])==0:
        df_status = pd.DataFrame({'X': [],'Y': []})
        print(PATH+'/device/light_loc_from_IR.csv')
        df_status.to_csv(PATH+'/device/light_loc_from_IR.csv',index=False)
        return df_status
    # write in to the CSV file
    
    #df_loc = pd.read_csv(PATH+'/light_loc_from_IR.csv')
    
    
    #for center in centers:
    X = [center[0] for center in centers]
    Y = [center[1] for center in centers]
    df_status = pd.DataFrame({'X': X,'Y': Y})
        #df_check['LED'+str(LED)] = [light_status]
    #df_check[s] = [datetime.datetime.now()]
    df_status.to_csv(PATH+'/device/light_loc_from_IR.csv',index=False)

# Simple draw label on an image; in our case, the video frame
def draw_label(cv_image, label_text, label_position):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255,255,255)
    # You can get the size of the string with cv2.getTextSize here
    cv2.putText(cv_image, label_text, label_position, font_face, scale, color, 1, cv2.LINE_AA)

# Read a frame from the camera, and draw the FPS on the image if desired
# Return an image

def read_camera(csi_camera,display_fps):
    _ , camera_image=csi_camera.read()
    if display_fps:
        draw_label(camera_image, "Frames Displayed (PS): "+str(csi_camera.last_frames_displayed),(10,20))
        draw_label(camera_image, "Frames Read (PS): "+str(csi_camera.last_frames_read),(10,40))
    return camera_image


DISPLAY_WIDTH=640#
DISPLAY_HEIGHT=480#

#DISPLAY_WIDTH = 3280
#DISPLAY_HEIGHT = 2464

width = DISPLAY_WIDTH
height = DISPLAY_HEIGHT 

# 1920x1080, 30 fps
SENSOR_MODE_1080=2
# 1280x720, 60 fps
SENSOR_MODE_720=3

capture_width=width
capture_height=height
framerate=3
rate = framerate
#flip_method,
display_width=width
display_height=height


b0=0



def gstreamer_pipeline(
    sensor_id=1,
    #ensor_mode=3,
    capture_width=width,
    capture_height=height,
    display_width=width,
    display_height=height,
    framerate=rate,
    #flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !" #sensor-mode=%d ! "
        ""
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=vertical-flip ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            #sensor_mode,
            capture_width,
            capture_height,
            framerate,
            #flip_method,
            display_width,
            display_height,
        )
    )




class MyVideoCapture:

    def __init__(self,sensor_id):
        
                  #try:
        self.sensor_id = sensor_id

        # store the light status for taking the frames 
        self.previous_light_status = ['off', 'off','off','off'] 

        self.bkgA = [None,None,None,None]
        self.bkgB = [None,None,None,None]
        self.imgA = [None,None,None,None]
        self.imgB = [None,None,None,None]

        if sensor_id >= 0:
            self.cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=sensor_id), cv2.CAP_GSTREAMER)

            


        if sensor_id == -1:
            print('running from file:')
            path = os.path.dirname(os.path.abspath(__file__))
            print('path: ', path)
            self.cap = cv2.VideoCapture(path+'/media/IR01.mp4')
            frame_width = int( self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height =int( self.cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
            ret, frame1 = self.cap.read()
            ret, self.frame1 = self.get_frame()

            person_scale= int(self.frame1.shape[0]/7)

        self.bkg0 = self.get_frame()
        self.bkg1 = self.get_frame()

        os.system("v4l2-ctl -c exposure=10")


    # Generate IR Frames for Manual Mode
    def get_frame(self):
        
        ret_val, img = self.cap.read()
        img = cv2.resize(img, (640,480), interpolation=cv2.INTER_AREA)

        # Creating a translation matrix
        translation_matrix = np.float32([ [1,0,-8], [0,1,8] ])

        # Image translation
        img = cv2.warpAffine(img, translation_matrix, (640,480))




        if ROTATE_CAM:
            img = cv2.rotate(img, cv2.ROTATE_180)

        img = cv2.resize(img, (640, 360))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        return ret_val, img
            


    # Generate IR Frames for Authomatic Mode and Find Centers of IR Spots
    def get_processed_IR_frame(self):

        #Capture Background Images when the All UVs are Off
        if True:
            _,self.frame1=self.get_frame()
            time.sleep(.100) #wait milliseconds before capturing the 2nd background frame
            _,self.frame2=self.get_frame()


            light_status = read_light_status()
            #print(light_status)
            #print(light_status[0], light_status[0]==False, light_status[0]==True)
            for i in range(4):
                #print(i)
                
                # for each LED update either background or the image
                if light_status[i]=='off':
                    self.bkgA[i]=self.frame1
                    self.bkgB[i]=self.frame2
                    
                    #print('self.bkgA[i], self.bkgB[i]',self.bkgA[i].shape, self.bkgB[i].shape)
                elif light_status[i]=='on':
                    self.imgA[i]=self.frame1
                    self.imgB[i]=self.frame2
                    #cv2.imwrite("frame_on.jpg", self.frame1)     # save frame as JPEG file

                    #print('self.imgA[i], self.imgB[i]',self.imgA[i].shape, self.imgB[i].shape)
                #print(light_status[i], self.previous_light_status[i])
                #print('status:', light_status[i], self.previous_light_status[i])
                if light_status[i]=='on' and self.previous_light_status[i]=='transition':
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    
                    # average the frames
                    bkg = cv2.addWeighted( self.bkgA[i], 0.5, self.bkgB[i], 0.5, 0.0)
                    img = cv2.addWeighted( self.imgA[i], 0.5, self.imgB[i], 0.5, 0.0)   # average of two images
                    PATH = os.path.dirname(os.path.abspath(__file__)) # path of the current file (not the master file)
                    print('saving image in', PATH+"/bkg.jpg")
                    cv2.imwrite(PATH+"/saved_images/"+timestr+"bkg.jpg", bkg)     # save frame as JPEG file
                    cv2.imwrite(PATH+"/saved_images/"+timestr+"img.jpg", img)     # save frame as JPEG file

                    img_diff = cv2.subtract(img, bkg)                    # Updated by moh
                    cv2.imwrite(PATH+"/saved_images/"+timestr+"img_diff.jpg", img_diff)
                    print(img_diff.shape)
                    #print(img_diff[100,100,0])
                    img_prcs = pre_process(img_diff)          # Process after subtraction
                    #print('img_prcs.shape: ', img_prcs.shape)
                    # get IR centers
                    
                    print('yes')
                    try:
                        centers, img = find_IR.find_loc(img_prcs)
                        print('find_IR.find_loc, centers', centers)
                        #img = annotate_centers(img,centers)
                    except:
                        print('ERROR: find_IR.find_loc(img_prcs)')
                        centers = [[]]

                    cv2.imwrite(PATH+"/saved_images/"+timestr+"img_prcs.jpg", img)
                    # update light_loc in the shared file
                    write_light_loc_from_IR(centers)
                    print('centers ', centers)

                    # update status
                self.previous_light_status[i] = light_status[i]

        if False:
                print('closing all windows')
                #self.cap.stop()
                self.cap.release()
                cv2.destroyAllWindows()


        #if ROTATE_CAM:
        #    self.frame1 = cv2.rotate(self.frame1, cv2.ROTATE_180)
        return True, self.frame1

    def close_all(self):
        self.cap.stop()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print('main')
    #run_camera(sensor_id=1, position=[100,100])