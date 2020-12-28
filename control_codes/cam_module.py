# Create a window and pass it to the Application object
import cv2


#sensor_id=0
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
    sensor_id=1,
    #ensor_mode=3,
    capture_width=capture_width,
    capture_height=capture_height,
    display_width=display_width,
    display_height=display_height,
    framerate=rate,
    #flip_method=0,
):
    #print('in function sensor_id: ', )
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
        print('sensor_id: ',sensor_id)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
            
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
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