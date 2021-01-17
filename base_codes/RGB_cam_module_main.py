# Create a window and pass it to the Application object
import cv2
import time
import datetime 
#import matplotlib.pyplot as plt

import numpy as np
from control_codes.person_detection.person_detection import person_detection
from control_codes.touch_detection.position_estimation_class_v2 import pose_estimation

sensor_id=0
capture_width = 640#1024
capture_height = 480#768
framerate = 20
#sensor_mode = 3
display_width = 640
display_height = 480

#width = capture_width
#height = capture_height
rate = framerate


def gstreamer_pipeline(
    sensor_id=0,
    #ensor_mode=3,
    capture_width=capture_width,
    capture_height=capture_height,
    display_width=display_width,
    display_height=display_height,
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



class MyVideoCapture:
    def __init__(self, sensor_id):
        # Open the video source
        #self.vid = cv2.VideoCapture(video_source)
        self.vid = cv2.VideoCapture(gstreamer_pipeline(sensor_id=sensor_id), cv2.CAP_GSTREAMER)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
            
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Load algorithm classes
        print('Loading classes...')
        self.my_person_detection = person_detection(person_scale=80)
        print('My_person_detection loaded')
        self.my_pose_estimation = pose_estimation()
        print('My pose loaded')


        ret, self.frame1 = self.get_frame()
        
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        #self.window.mainloop()
        
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # apply detections
    def get_processed_frame(self):
        ret, frame2 = self.get_frame()
        if ret:

            # Detect events
            moving_people, moving_areas, still_centers = self.my_person_detection.get_all_people(self.frame1,frame2)
            
            # Touch Detection
            for box in moving_people:
                h,w = frame2.shape[0], frame2.shape[1] # rows and columns
                x1,x2 = max(min(box[0],box[2]),0), min(max(box[0],box[2]),w)
                y1,y2 = max(min(box[1],box[3]),0), min(max(box[1],box[3]),h)
                cv2.rectangle(frame2, (x1,y1),(x2,y2), (0, 255, 0), 4)

                if box[5]<150 and x2-x1>0 and y2-y1>0 and x1<w and y1<h: # if the distance from the previous box is small and ...
                    cv2.putText(frame2, 'looking for touch', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3)
                    print('looking for touch')
                    print(x1,x2,y1,y2, frame2.shape)

                    frame_region = frame2[y1:y2,x1:x2,:]
                    print('**********',frame_region.shape)
                    
                    frame_region, touch_spots = self.my_pose_estimation.detect_touch(frame_region)
                    print('**',frame_region.shape)
                    frame2[y1:y2,x1:x2,:] = frame_region

                    ### Touch locations:

            #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            
            # Show and write the image
            #image = cv2.resize(frame1, (1280,720))
            #image = cv2.resize(frame1, (600,400))
            #st = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            #print('main.py: datetime:', st)

            #out.write(cv2.resize(frame1, (1280,720)))
            #cv2.imshow("feed", cv2.resize(frame1, (600,400)))
            self.frame1 = frame2
        return ret, frame2
