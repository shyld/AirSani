# MIT License
# Copyright (c) 2019 JetsonHacks
# See LICENSE for OpenCV license and additional information

# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV

import cv2
import numpy as np
from csi_camera import CSI_Camera
import time
import datetime
from find_IR import find_light
import os
from multiprocessing import Process


#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)


show_fps = True

# Simple draw label on an image; in our case, the video frame
def draw_label(cv_image, label_text, label_position):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255,255,255)
    # You can get the size of the string with cv2.getTextSize here
    cv2.putText(cv_image, label_text, label_position, font_face, scale, color, 1, cv2.LINE_AA)

# Read a frame from the camera, and draw the FPS on the image if desired


# Good for 1280x720
DISPLAY_WIDTH=640
DISPLAY_HEIGHT=360
# For 1920x1080
# DISPLAY_WIDTH=960
# DISPLAY_HEIGHT=540

# 1920x1080, 30 fps
SENSOR_MODE_1080=2
# 1280x720, 60 fps
SENSOR_MODE_720=3

b0=0

sensor_id=0
capture_width = 1280
capture_height = 720
framerate = 20
sensor_mode = 3
display_width = 640
display_height = 360

width = capture_width
height = capture_height
rate = framerate

def gstreamer_pipeline(
    sensor_id=0,
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


# preprocessing
def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (width,height))
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    return img

class camera_obj:

    def __init__(self):
        #self.cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
        self.cap = cv2.VideoCapture('media/01.mp4')
        self.frame_width = int( self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height =int( self.cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

        # Start counting the number of frames read and displayed
        #left_camera.start_counting_fps()
    def get_frame(self):
        
        try:
            ret_val, img = self.cap.read()
            if ret_val:
                print(img.mean(axis=0).mean(axis=0))
            else:
                print('No image read')

        finally:
            cv2.destroyAllWindows()
            self.cap.release()
            
        return img
            



if __name__ == "__main__":
    print('main run')
    #run_camera(sensor_id=1, position=[100,500])