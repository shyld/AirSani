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
# Return an image
def read_camera(csi_camera,display_fps):
    _ , camera_image=csi_camera.read()
    if display_fps:
        draw_label(camera_image, "Frames Displayed (PS): "+str(csi_camera.last_frames_displayed),(10,20))
        draw_label(camera_image, "Frames Read (PS): "+str(csi_camera.last_frames_read),(10,40))
    return camera_image

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

# preprocessing
def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (width,height))
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    return img


def run_camera(sensor_id, position):

    left_camera = CSI_Camera()
    left_camera.create_gstreamer_pipeline(
            sensor_id=sensor_id,
            sensor_mode=SENSOR_MODE_720,
            framerate=30,
            flip_method=0,
            display_height=DISPLAY_HEIGHT,
            display_width=DISPLAY_WIDTH,
    )
    left_camera.open(left_camera.gstreamer_pipeline)
    left_camera.start()
    cv2.namedWindow('IR Image', cv2.WINDOW_AUTOSIZE)

    if (
        not left_camera.video_capture.isOpened()
     ):
        # Cameras did not open, or no camera attached

        print("Unable to open any cameras")
        # TODO: Proper Cleanup
        SystemExit(0)

  #try:
    img0=read_camera(left_camera,False)
    try:

        # Get the initial frame
        for i in range(40):
            img0=read_camera(left_camera,False)

        cv2.imshow('IR Image',img0) 
        cv2.moveWindow('IR Image', position[0], position[1])

        img0 = pre_process(img0)
        #print(img0.shape)

        # Start counting the number of frames read and displayed
        left_camera.start_counting_fps()
        b0=0
        while True:
            img1=read_camera(left_camera,False)
            img1 = pre_process(img1)
            img = cv2.absdiff(img0, img1)
                # the diff img
                
            #ret_val, img1 = cap.read()
            t = datetime.datetime.now()
            b1 =  int(t.second*10)
            if b1 != b0:
                b0 = b1

                try:
                    centers,img = find_light.find_loc(img)
                    center = centers[0]
                    cx,cy = int(center[0]-img.shape[1]/2), int(img.shape[0]/2 - center[1])
                    cv2.putText(img, "("+str(cx)+","+str(cy)+")", (1000, 1000), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 5, cv2.LINE_AA)
                except:
                    img=img
            cv2.imshow('IR Image',img)
            if cv2.waitKey(10) == 27:
                break
            
    finally:
    #if False:
        left_camera.stop()
        left_camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera(sensor_id=1, position=[100,500])