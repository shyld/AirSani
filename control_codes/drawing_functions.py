# MIT License
# Copyright (c) 2019 JetsonHacks
# See LICENSE for OpenCV license and additional information

# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV
import pandas as pd
import cv2
import numpy as np
import time
import datetime

from control_codes.person_detection.person_detection import person_detection


from sklearn.neighbors import NearestNeighbors

from control_codes import shared_variables



# Simple draw label on an image; in our case, the video frame
def draw_label(cv_image, label_text, label_position):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255,255,255)
    # You can get the size of the string with cv2.getTextSize here
    cv2.putText(cv_image, label_text, label_position, font_face, scale, color, 1, cv2.LINE_AA)

def draw_detections(frame):
    C = shared_variables.detected_coordinates
    #print('draw_detections ', len(C))
    for i in range(len(C)):
        #print("C.loc[i,'Left']: ",C.loc[i,'Left'])
        #print("C.loc[i,'Left']: ",C.iloc[i,'Left'])
        #print("C.loc[i,'Left']: ",C['Left'].iloc[i])

        x1,x2,y1,y2 = int(C['Left'].iloc[i]), int(C['Right'].iloc[i]), int(C['Top'].iloc[i]), int(C['Bottom'].iloc[i])
        alpha = 0.7
        frame = overlay_square(frame, x1,y1,x2,y2,(255,0,0),alpha)

    return frame


def overlay_square(frame, x1,y1,x2,y2,color,alpha):
    overlay = frame.copy()
    output = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2),color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
    return output

def draw_scores(frame):
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
        frame = overlay_square(frame, x1,y1,x2,y2,(255,0,0),alpha)
            #sub_img = frame[y1:y2, x1:x2]
            #white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

            #res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        # Putting the image back to its position
            #frame[y1:y2, x1:x2] = res

        #cv2.putText(frame, "("+str(sc)+")", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,0,0), 5, cv2.LINE_AA)


    return frame

def get_random_UV_loc(X):
    R = []
    if X.shape[0]<4:
        return R

    len_df = len(shared_variables.scored_spots)

    
    u=0
    while len(R)<4 and u<20:
        u+=1
        r = np.random.randint(low=0,high=len_df,size=1)[0]
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X)
        x,y = shared_variables.scored_spots.loc[r,'i'],shared_variables.scored_spots.loc[r,'j']
        D,I = neigh.kneighbors([[x,y]], n_neighbors=2, return_distance=True)
        #print('################. X',X)
        #print('################. x,y',x,y)
        #print('################. D',D)
        if np.sum(D[0,0]<100)==0:
            R.append(r)

    return R

def apply_UV():
    R=[]
    if len(shared_variables.scored_spots)<4:
        return 0



    t = datetime.datetime.now()
    t_s =  int(t.second/4)
    if t_s!= shared_variables.tf_UV:
        shared_variables.tf_UV = t_s

        # clear previous spots
        df_filter = shared_variables.scored_spots[shared_variables.scored_spots['score']==-1]
        #print('LEN 1', len(shared_variables.scored_spots))
        shared_variables.scored_spots = shared_variables.scored_spots[shared_variables.scored_spots['score']>0].reset_index(drop=True)
        #print('LEN 2', len(shared_variables.scored_spots))
        

        # Assign new spots
        #df_score = shared_variables.scored_spots
        #df_score = df_score.reset_index(drop=True)

        len_df = len(shared_variables.scored_spots)
        #print('in APPLY UV', len_df)
        # load the dataset
        # randomly select 4
        if len_df ==0:
            return 0

        X1 = np.reshape(shared_variables.detected_coordinates['Left'].to_numpy(),(-1,1))+np.reshape(shared_variables.detected_coordinates['Right'].to_numpy(),(-1,1))
        X2 = np.reshape(shared_variables.detected_coordinates['Top'].to_numpy(),(-1,1))+np.reshape(shared_variables.detected_coordinates['Bottom'].to_numpy(),(-1,1))
        X1 = (X1/2 - shared_variables.Cam_width/2).astype(int)
        X2 = (X2/2 - shared_variables.Cam_height/2).astype(int)

        #print('X1.shape', X1.shape)
        X = np.concatenate((X1,X2),axis=1)
        
        #print('X.shape', X.shape)
        
        R = get_random_UV_loc(X)


        print(R, len_df)
        shared_variables.UV_spots = R

    # change the scores to -1
    if True:
        #df_score = shared_variables.scored_spots

        for i in range(len(R)):
            shared_variables.scored_spots.loc[shared_variables.UV_spots[i],'score']=-1
            #print('df_score.loc[R[i]]',shared_variables.scored_spots.loc[shared_variables.UV_spots[i]])
    #except:
        #    print('ERRROR IN INDEX')

        #print('in APPLY UV >>>>>>>>>',df_score[df_score.score==-1])

    #return df_score

def draw_UV( frame):

    try:
        shared_variables.read_scores_from_file()
        #C = shared_variables.scored_spots
        #print('in score_df try', len(C))
        #df_new =  pd.concat([df, detected_coordinates])
    except:
        #C = pd.DataFrame({'time':[], 'priority':[],'i':[], 'j':[],'score':[]})
        print('in score_df except')


    apply_UV()

    C = shared_variables.scored_spots

    df_filter = C[C['score']==-1]

    #print('df_filter :', df_filter)
    
    #print('IN DRAW UV', idx)
    if len(df_filter)==0:
        #print('ZERO -1 LEN')
        return frame

    df_UV = C[C['score']==-1]

    #print('00000000      df_UV len', df_UV)

    #print(df_score[df_score['score']==-1])

    #print('df_score: ', df_score)
    if len(df_UV)>0:
        #print('111111111     df_UV len', len(df_UV))
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

            #print('^^^^^^^ UV Apply ^^^^^^: ', x1,x2,y1,y2) 
            alpha = 0.7
            frame = overlay_square(frame, x1,y1,x2,y2,(0,255,0),alpha)
    return frame

def draw_wall(self,frame):
    X = shared_variables.still_people
    if X.shape[0]<2:
        return frame
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(X)
        
    D,I = neigh.kneighbors(X, n_neighbors=2, return_distance=True)
    idx = np.where((D[:,1]<300) & (D[:,1]>80))[0]
    if idx.shape[0]>0:
        p1 = X[idx[0],:]
        p2 = X[I[idx[0],1],:]
        print('p1,p2', p1,p2)
        alpha = 0.7
        frame = overlay_square(frame, x1,y1,x2,y2,(0,255,0),alpha)
    return frame

