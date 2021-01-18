import os
#print('running F1')
#os.system('python3 detection.py')
#print('running F2')
#os.system('python3 priority.py')
import subprocess

path = os.path.dirname(os.path.abspath(__file__))

print(path)

#subprocess.run("python3 main_all.py & python3 control_codes/process_detections.py", shell=True)
#  & sudo python3 control_codes/monitor.py
print(path+"/control_codes/process_detections.py")

#with open(path+"/control_codes/shared_csv_files/F2_log.csv","wb") as out, open(path+"/control_codes/shared_csv_files/F2_log_err.csv","wb") as err:
subprocess.Popen(["python3 "+ path+"/control_codes/process_detections.py"], close_fds=True, shell=True)



