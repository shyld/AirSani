# MIT License
# Copyright (c) 2019 JetsonHacks
# See LICENSE for OpenCV license and additional information

# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import cv2
import numpy as np
##from csi_camera import CSI_Camera
import time
import datetime
from find_IR import find_loc
##import shared_variables

##show_fps = True

# Simple draw label on an image; in our case, the video frame
#def draw_label(cv_image, label_text, label_position):
#    font_face = cv2.FONT_HERSHEY_SIMPLEX
#    scale = 0.5
#    color = (255,255,255)
    # You can get the size of the string with cv2.getTextSize here
#    cv2.putText(cv_image, label_text, label_position, font_face, scale, color, 1, cv2.LINE_AA)

# Read a frame from the camera, and draw the FPS on the image if desired
# Return an image

#def read_camera(csi_camera,display_fps):
#    _ , camera_image=csi_camera.read()
#    if display_fps:
#        draw_label(camera_image, "Frames Displayed (PS): "+str(csi_camera.last_frames_displayed),(10,20))
#        draw_label(camera_image, "Frames Read (PS): "+str(csi_camera.last_frames_read),(10,40))
#    return camera_image


# Good for 1280x720
DISPLAY_WIDTH=640
DISPLAY_HEIGHT=480
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




#class MyVideoCapture:

#    def __init__(self,sensor_id):
#        try:
            #sensor_id = 1
#            self.left_camera = CSI_Camera()
#            self.left_camera.create_gstreamer_pipeline(
#                    sensor_id=sensor_id,
#                    sensor_mode=SENSOR_MODE_720,
#                    framerate=30,
#                    flip_method=0,
#                    display_height=DISPLAY_HEIGHT,
#                    display_width=DISPLAY_WIDTH,
#            )
#            self.left_camera.open(self.left_camera.gstreamer_pipeline)
#            self.left_camera.start()
            #cv2.namedWindow('IR Image', cv2.WINDOW_AUTOSIZE)

#            if (
#                not self.left_camera.video_capture.isOpened()
#             ):
                # Cameras did not open, or no camera attached

#                print("Unable to open any cameras")
                # TODO: Proper Cleanup
#                SystemExit(0)

          #try:

#            self.bkg0 = self.get_frame()
#            self.bkg1 = self.get_frame()

            #img0=read_camera(self.left_camera,False)

#        except:
#            print('finally: init')
#            self.left_camera.stop()
#            self.left_camera.release()
#            cv2.destroyAllWindows()



    # Generate IR Frames for Manual Mode
 #   def get_IR_frame(self):
 #       try:
 #           img = read_camera(left_camera,False)
 #       except:
    #if False:
 #           print('finally: get_frame')
 #           self.left_camera.stop()
 #           self.left_camera.release()
 #           cv2.destroyAllWindows()
            
 #       return True, img


    # Generate IR Frames for Authomatic Mode and Find Centers of IR Spots
#def get_processed_IR_frame(self):
        #bkg0=read_camera(left_camera,False)
#        try:

            #Capture Background Images when the All UVs are Off
#            if shared_variables.UV_Active = False
#                self.bkg0=read_camera(left_camera,False) # Background Image Updated by moh
                #time.sleep(.100) #wait milliseconds before capturing the 2nd background frame
#                self.bkg1=read_camera(left_camera,False) # Background Image Updated by moh
#                IR_frame = self.bkg0

            # Test Images From Files
height,width = 480,640
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray=cv2.resize(gray, (height, width))

#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image=cv2.resize(image, (width, height))

bkg1 = cv2.resize(cv2.imread('BKGA1.jpg',0), (width, height), interpolation = cv2.INTER_AREA)
bkg0 = cv2.resize(cv2.imread('BKGA0.jpg',0), (width, height), interpolation = cv2.INTER_AREA)


        #cv2.imshow('IR Image',bkg0) 
        #cv2.moveWindow('IR Image', position[0], position[1])

        #Capture IR Spots when the at least one UV is on
        #if shared_variables.UV_Active = True
#bkg0 = cv2.cvtColor(bkg0, cv2.COLOR_BGR2GRAY)       # added by moh
#bkg1 = cv2.cvtColor(bkg1, cv2.COLOR_BGR2GRAY)       # added by moh

bkg = cv2.addWeighted( bkg0, 0.5, bkg1, 0.5, 0.0)   # average of two images - added by moh


            # Start counting the number of frames read and displayed
#                left_camera.start_counting_fps()
#                b0=0
            #while True:
#                img0 = read_camera(left_camera,False)             # added by moh
#                time.sleep(.100) #wait milliseconds before capturing the 2nd background frame
#                img1 = read_camera(left_camera,False)             # added by moh

           # Test Images From Files
img1 = cv2.resize(cv2.imread('IMGA1.jpg',0), (width, height), interpolation = cv2.INTER_AREA)
img0 = cv2.resize(cv2.imread('IMGA0.jpg',0), (width, height), interpolation = cv2.INTER_AREA)

plt.imshow(img0)
plt.show()

#img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)   # added by moh
#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)   # added by moh         



img = cv2.addWeighted( img0, 0.5, img1, 0.5, 0.0) # average of two images - added by moh
img = cv2.subtract(img, bkg)                    # Updated by moh
img = pre_process(img)          # Process after subtraction

            #cv2.imshow('IR Image',img)
            #cv2.imshow('IR Image',img)
            #cv2.waitKey(0)
                    # the diff img
                    
                #ret_val, img1 = cap.read()
#                t = datetime.datetime.now()
#                b1 =  int(t.second*10)
#                if b1 != b0:
#                    b0 = b1

#try:
print(img.shape)
centers,img = find_loc(img)
print(img.shape)

plt.imshow(img)
plt.show()
center = centers[0]
print(centers)
#plt.imshow(img)
#plt.show()
#print(centers)
cx,cy = int(center[0]-img.shape[1]/2), int(img.shape[0]/2 - center[1])
#print(cx,cy)
# Delete centers that are outside the lens circle -> Eliminate Reflections from the Edge of AirSani
#cx,cy = [ [x for x in cx], [y for y in cy] if (math.sqrt((x2**2) + (y**2))) <= 0.9*(img.shape[0]**2)/4 ]
#shared_values.IR_Centers = centers
# Add Text to the Centers on the Raw IR Image 
cv2.putText(img0, "("+str(cx)+","+str(cy)+")", (1000, 1000), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 5, cv2.LINE_AA)
#except:
#    img0=img0


                #if cv2.waitKey(10) == 27:
                #    break
#            IR_frame = img0
                
#        finally:
        #if False:
#            left_camera.stop()
#            left_camera.release()
#            cv2.destroyAllWindows()

#        return IR_frame


#if __name__ == "__main__":
    #run_camera(sensor_id=1, position=[100,100])
