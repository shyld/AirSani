import cv2
import os, signal
from multiprocessing import Process
import subprocess
from control_codes import shared_variables
import datetime
from control_codes.RGB_cam_module import MyVideoCapture
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))
#print('path: ', path)
print('shared_variables.TEST', shared_variables.TEST)
print(path+"/control_codes/process_detections.py")

#with open(path+"/control_codes/shared_csv_files/F2_log.csv","wb") as out, open(path+"/control_codes/shared_csv_files/F2_log_err.csv","wb") as err:
subprocess.Popen(["python3 "+ path+"/control_codes/process_detections.py"],close_fds=True, shell=True) # stdout=out, stderr=err, 


#out, err = p.communicate()  # This will get you output
#print('err', out, err)

vid = MyVideoCapture(sensor_id=-1)

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720)) # width, height

# A function to check the stopping rule
a0,a1, b0, b1, c0,c1 = 0,0,0,0,0,0

def check_stop(b0,b1):
	F_running = True
	if b1 != b0:
		print('Check_stop')
		b0 = b1
		df0 = pd.read_csv(path+'/control_codes/shared_csv_files/onoff.csv')
		F_running = (df0[df0['arguments']=='F1']['value'].iloc[0]=='TRUE') & (df0[df0['arguments']=='F0']['value'].iloc[0]=='TRUE')
	return b0, F_running

F_running = True

while F_running:

	t = datetime.datetime.now()
	b1 =  int(t.second)
	b0, F_running = check_stop(b0,b1)

	#print('shared_variables.TEST', shared_variables.TEST)
	#try:
	if True:
		ret, frame = vid.get_processed_frame()
		#print('##### ORIGINAL FRAME SIZE #####: ', frame.shape)

		out.write(cv2.resize(frame, (1280,720)))
		cv2.imshow("feed", cv2.resize(frame, (600,400)))

		if cv2.waitKey(100) == 27:
			break
	#except:
	#	os.killpg(os.getpid(), signal.SIGTERM)

os.killpg(os.getpid(), signal.SIGTERM)