import os
#print('running F1')
#os.system('python3 detection.py')
#print('running F2')
#os.system('python3 priority.py')
import subprocess


subprocess.run("python3 main_all.py & python3 control_codes/process_detections.py", shell=True)
#  & sudo python3 control_codes/monitor.py
