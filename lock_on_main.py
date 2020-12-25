import time
import os 

while True:
	time.sleep(2)
	#print('lock')
	os.popen('if test $(wmctrl -l | grep "Wi-Fi Network Authentication Required" 2>&1 | wc -l) -eq 1; then wmctrl -a "Wi-Fi Network Authentication Required"; elif test $(wmctrl -l | grep "Network" 2>&1 | wc -l) -eq 1; then wmctrl -a "Network"; else wmctrl -a "Shyld AI"; fi')