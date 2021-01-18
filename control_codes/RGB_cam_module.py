# MIT License
# Copyright (c) 2019 JetsonHacks
# See LICENSE for OpenCV license and additional information

# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV
import pandas as pd
import cv2
import numpy as np
from control_codes.csi_camera import CSI_Camera
import time
import datetime
#from find_IR import find_light
import os
from multiprocessing import Process

from control_codes.person_detection.person_detection import person_detection
from control_codes.touch_detection.position_estimation_class_v2 import pose_estimation
from sklearn.neighbors import NearestNeighbors

from control_codes import shared_variables
shared_variables.init()
#print('shared.variables.avoid_list', shared_variables.avoid_list)

PATH = os.path.dirname(os.path.abspath(__file__))

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
#DISPLAY_WIDTH=640
#DISPLAY_HEIGHT=360
# For 1920x1080
DISPLAY_WIDTH=640#3280
DISPLAY_HEIGHT=480#2464

#DISPLAY_WIDTH = 3280
#DISPLAY_HEIGHT = 2464

width = DISPLAY_WIDTH
height = DISPLAY_HEIGHT 

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

class MyVideoCapture:

    def __init__(self,sensor_id):
        
        try:
            #sensor_id = 1
            self.left_camera = CSI_Camera()
            self.left_camera.create_gstreamer_pipeline(
                    capture_width=width,
                    capture_height=height,
                    sensor_id=sensor_id,
                    #sensor_mode=SENSOR_MODE_1080,
                    framerate=5,
                    flip_method=0,
                    display_height=DISPLAY_HEIGHT,
                    display_width=DISPLAY_WIDTH,
            )
            self.left_camera.open(self.left_camera.gstreamer_pipeline)
            self.left_camera.start()
            cv2.namedWindow('IR Image', cv2.WINDOW_AUTOSIZE)

            if (
                not self.left_camera.video_capture.isOpened()
             ):
                # Cameras did not open, or no camera attached

                print("Unable to open any cameras")
                # TODO: Proper Cleanup
                SystemExit(0)

          #try:
            ret, self.frame1 = self.get_frame()
            #print('self.frame1.shape', self.frame1.shape)

            person_scale= int(self.frame1.shape[0]/7)

            # Load algorithm classes
            print('Loading classes...')
            self.my_person_detection = person_detection(person_scale=person_scale)
            print('My_person_detection loaded')
            self.my_pose_estimation = pose_estimation()
            print('My pose loaded')



            #img0=read_camera(self.left_camera,False)

        except:
            print('finally: init')
            self.left_camera.stop()
            self.left_camera.release()
            cv2.destroyAllWindows()




            #img0=read_camera(self.left_camera,False)

        #except:
        #    print('finally: init')
        #    cv2.destroyAllWindows()


        # Get the initial frame
        #for i in range(40):
        #    img0=read_camera(self.left_camera,False)

        #cv2.imshow('IR Image',img0) 
        #cv2.moveWindow('IR Image', position[0], position[1])

        #self.img_ref = pre_process(img0)
        #print(img0.shape)

        # Start counting the number of frames read and displayed
        #left_camera.start_counting_fps()
    def draw_detections(self,frame):
        C = shared_variables.detected_coordinates
        #print('draw_detections ', len(C))
        for i in range(len(C)):
            #print("C.loc[i,'Left']: ",C.loc[i,'Left'])
            #print("C.loc[i,'Left']: ",C.iloc[i,'Left'])
            #print("C.loc[i,'Left']: ",C['Left'].iloc[i])

            x1,x2,y1,y2 = int(C['Left'].iloc[i]), int(C['Right'].iloc[i]), int(C['Top'].iloc[i]), int(C['Bottom'].iloc[i])
            alpha = 0.7
            frame = self.overlay_square(frame, x1,y1,x2,y2,(0,0,255),alpha)

        return frame


    def overlay_square(self, frame, x1,y1,x2,y2,color,alpha):
        overlay = frame.copy()
        output = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2),color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
        return output

    def draw_scores(self,frame):
        #shared_variables.read_scores_from_file()
        C = shared_variables.scored_spots

        print('********************** draw_scores ', len(C))
        for i in range(len(C)):
            #print("C.loc[i,'Left']: ",C.loc[i,'Left'])
            #print("C.loc[i,'Left']: ",C.iloc[i,'Left'])
            #print("C.loc[i,'Left']: ",C['Left'].iloc[i])

            t,pr,x,y,sc = C['time'].iloc[i], int(C['priority'].iloc[i]), int(C['i'].iloc[i]), int(C['j'].iloc[i]), int(C['score'].iloc[i])
            #print('detection: ', x1,x2,y1,y2) 
            y1=y+ int(shared_variables.Cam_height/2)
            y2=y+ int(shared_variables.Cam_height/2 + shared_variables.Coverage_size)
            x1=x + int(shared_variables.Cam_width/2)
            x2=x + int(shared_variables.Cam_width/2 +shared_variables.Coverage_size)

            alpha = 0.3
            frame = self.overlay_square(frame, x1,y1,x2,y2,(0,0,255),alpha)
                #sub_img = frame[y1:y2, x1:x2]
                #white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

                #res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            # Putting the image back to its position
                #frame[y1:y2, x1:x2] = res

            #cv2.putText(frame, "("+str(sc)+")", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,0,0), 5, cv2.LINE_AA)


        return frame

    def get_random_UV_loc(self,X):
        len_df = len(shared_variables.scored_spots)

        R = []
        u=0
        while len(R)<4 and u<20:
            u+=1
            r = np.random.randint(low=0,high=len_df,size=1)[0]
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(X)
            x,y = shared_variables.scored_spots.loc[r,'i'],shared_variables.scored_spots.loc[r,'j']
            D,I = neigh.kneighbors([[x,y]], n_neighbors=2, return_distance=True)
            print('################. X',X)
            print('################. x,y',x,y)
            print('################. D',D)
            if np.sum(D[0,0]<100)==0:
                R.append(r)

        return R

    def apply_UV(self):
        R=[]
        if len(shared_variables.scored_spots)<4:
            return 0



        t = datetime.datetime.now()
        t_s =  int(t.second/4)
        if t_s!= shared_variables.tf_UV:
            shared_variables.tf_UV = t_s

            # clear previous spots
            df_filter = shared_variables.scored_spots[shared_variables.scored_spots['score']==-1]
            print('LEN 1', len(shared_variables.scored_spots))
            shared_variables.scored_spots = shared_variables.scored_spots[shared_variables.scored_spots['score']>0].reset_index(drop=True)
            print('LEN 2', len(shared_variables.scored_spots))
            

            # Assign new spots
            #df_score = shared_variables.scored_spots
            #df_score = df_score.reset_index(drop=True)

            len_df = len(shared_variables.scored_spots)
            print('in APPLY UV', len_df)
            # load the dataset
            # randomly select 4
            if len_df ==0:
                return 0

            X1 = np.reshape(shared_variables.detected_coordinates['Left'].to_numpy(),(-1,1))+np.reshape(shared_variables.detected_coordinates['Right'].to_numpy(),(-1,1))
            X2 = np.reshape(shared_variables.detected_coordinates['Top'].to_numpy(),(-1,1))+np.reshape(shared_variables.detected_coordinates['Bottom'].to_numpy(),(-1,1))
            X1 = (X1/2 - shared_variables.Cam_width/2).astype(int)
            X2 = (X2/2 - shared_variables.Cam_height/2).astype(int)

            print('X1.shape', X1.shape)
            X = np.concatenate((X1,X2),axis=1)
            print('X.shape', X.shape)
            
            R = self.get_random_UV_loc(X)


            print(R, len_df)
            shared_variables.UV_spots = R

        # change the scores to -1
        if True:
            #df_score = shared_variables.scored_spots

            for i in range(len(R)):
                shared_variables.scored_spots.loc[shared_variables.UV_spots[i],'score']=-1
                print('df_score.loc[R[i]]',shared_variables.scored_spots.loc[shared_variables.UV_spots[i]])
        #except:
            #    print('ERRROR IN INDEX')

            #print('in APPLY UV >>>>>>>>>',df_score[df_score.score==-1])

        #return df_score

    def draw_UV(self, frame):

        try:
            shared_variables.read_scores_from_file()
            #C = shared_variables.scored_spots
            #print('in score_df try', len(C))
            #df_new =  pd.concat([df, detected_coordinates])
        except:
            #C = pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})
            print('in score_df except')


        self.apply_UV()

        C = shared_variables.scored_spots

        df_filter = C[C['score']==-1]

        #print('df_filter :', df_filter)
        
        #print('IN DRAW UV', idx)
        if len(df_filter)==0:
            print('ZERO -1 LEN')
            return frame

        df_UV = C[C['score']==-1]

        #print('00000000      df_UV len', df_UV)

        #print(df_score[df_score['score']==-1])

        #print('df_score: ', df_score)
        if len(df_UV)>0:
            print('111111111     df_UV len', len(df_UV))
            # remove the indexes
            
            #df_score  = df_score[idx].drop().reset_index(drop=True)
            #df_score.to_csv(PATH+'/shared_csv_files/scored_spots.csv',index=False)
    
            # plot 
            for i in range(len(df_UV)):

                t,pr,x,y,sc = df_filter['time'].iloc[i], int(df_filter['priority'].iloc[i]), int(df_filter['i'].iloc[i]), int(df_filter['j'].iloc[i]), int(df_filter['score'].iloc[i])
                
                y1=y+ int(shared_variables.Cam_height/2)
                y2=y+ int(shared_variables.Cam_height/2 + shared_variables.Coverage_size)
                x1=x + int(shared_variables.Cam_width/2)
                x2=x + int(shared_variables.Cam_width/2 +shared_variables.Coverage_size)

                print('^^^^^^^ UV Apply ^^^^^^: ', x1,x2,y1,y2) 
                alpha = 0.7
                frame = self.overlay_square(frame, x1,y1,x2,y2,(0,255,0),alpha)
        return frame






    def get_frame(self):
        
        try:
            #self.left_camera.start_counting_fps()
            img=read_camera(self.left_camera,False)
            #print('RGB: ',img.shape[0])
            #img1 = pre_process(img1)
            #img = cv2.absdiff(self.img_ref, img1)
            #print(img.shape)
            #centers,img = find_light.find_loc(img)
            #center = centers[0]
            #cx,cy = int(center[0]-img.shape[1]/2), int(img.shape[0]/2 - center[1])
            #cv2.putText(img, "("+str(cx)+","+str(cy)+")", (1000, 1000), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 5, cv2.LINE_AA)
        except:
    #if False:
            print('finally: get_frame')
            self.left_camera.stop()
            self.left_camera.release()
            cv2.destroyAllWindows()


            
        return True, img
            

    # apply detections
    def get_processed_frame(self):
        # Resize the Frame;
        #frame= cv2.resize(frame, (640,480))

        ret, self.frame1 = self.get_frame()
        time.sleep(0.1)
        ret, frame2 = self.get_frame()
        #print('in get_processed_frame: frame1.shape, frame2.shape: ', frame1.shape, frame2.shape)
        #print('in get prosessed frame, ', ret)
        if ret:
            # Detect events
            moving_people, moving_areas, still_centers, M = self.my_person_detection.get_all_people(self.frame1,frame2)
            #print('RGB_cam: moving_areas.shape ', moving_areas.shape)
            # Update Frame 1
            #self.frame1 = frame2
            #print('moving_people: ', moving_people)
            #print('moving_areas: ', moving_areas)

            # Add the detections to the shared_variables
            shared_variables.add_detections(moving_areas, priority=2)
            
            #print('in RGB_cam, shared_variables.detected_coordinates', len(shared_variables.detected_coordinates))
            #print('len(shared_variables.detected_coordinates) ',len(shared_variables.detected_coordinates))
            
            #shared_variables.remove_old_detection_list()
            #print('shared_variables.detected_coordinates', shared_variables.detected_coordinates)
            #print('shared_variables.scored_spots', shared_variables.scored_spots)
            #shared_variables.remove_old_scored_list()
            #print('shared_variables.scored_spots', shared_variables.scored_spots)
        #shared_variables.add_detections(moving_areas, priority=2)
        #shared_variables.remove_old_detection_list()
            #shared_variables.add_still_people(still_centers)
            
            # Update the avoid list:
            #shared_variables.add_avoid_list(moving_areas)
            #shared_variables.add_avoid_list(moving_people)
            #shared_variables.remove_old_avoid_list()

            # Touch Detection
            #print('in RGB_cam: moving_people', moving_people.shape)
            if True:
                for box in moving_people:
                    h,w = frame2.shape[0], frame2.shape[1] # rows and columns
                    x1,x2 = max(min(box[0],box[2]),0), min(max(box[0],box[2]),w)
                    y1,y2 = max(min(box[1],box[3]),0), min(max(box[1],box[3]),h)
                    cv2.rectangle(frame2, (x1,y1),(x2,y2), (0, 255, 0), 4)

                    if box[5]<100 and x2-x1>0 and y2-y1>0 and x1<w and y1<h: # if the distance from the previous box is small and ...
                        #cv2.putText(frame2, 'looking for touch', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 2)
                        #print('looking for touch')
                        #print(x1,x2,y1,y2, frame2.shape)

                        frame_region = frame2[y1:y2,x1:x2,:]
                        #print('**********',frame_region.shape)
                        
                        #frame_region, touch_spots = self.my_pose_estimation.detect_touch(frame_region)
                        
                        # Update shared variables
                        # if touch_spots.shape[0]>0:
                        #    shared_variables.add_detections(touch_spots, priority=1)
                        #print('**',frame_region.shape)
                        frame2[y1:y2,x1:x2,:] = frame_region

                
            # Draw the infected areas
            print('CHECK 1',len(shared_variables.scored_spots))
            frame2 = self.draw_detections(frame2)
            print('CHECK 2',len(shared_variables.scored_spots))
            frame2 = self.draw_scores(frame2)
            print('CHECK 3',len(shared_variables.scored_spots))
            frame2 = self.draw_UV(frame2)
            print('CHECK 4',len(shared_variables.scored_spots))

            ## Write to the file
            shared_variables.write_detections_to_file()
            print('CHECK 5',len(shared_variables.scored_spots))

            

        return ret, frame2


if __name__ == "__main__":
    print('main run')
    #run_camera(sensor_id=1, position=[100,500])