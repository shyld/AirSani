# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:01:04 2020

@author: Administrator
"""

import tkinter
#import numpy
from tkinter import *
import numpy as np
from tkinter import ttk

import PIL

from PIL import Image, ImageTk

import datetime
import cv2
import os,signal
from multiprocessing import Process
import subprocess

global program_mode
program_mode = 'off'


import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/shyldai/shyld/product_test')
#sys.path.insert(1, './control_codes')
#sys.path.insert(1, './control_codes/device')
#sys.path.insert(1, './control_codes/cam_IR')
sys.path.insert(1, '/home/shyldai/shyld/AiSani')
sys.path.insert(1, '/home/shyldai/shyld/AiSani/control_codes')
sys.path.insert(1, '/home/shyldai/shyld/AiSani/control_codes/device')

#from cam_IR import run_camera
from control_codes.device import steer
from control_codes.RGB_cam_module import MyVideoCapture
from control_codes.IR_cam_module import MyVideoCapture as MyIRCapture
#from control_codes.cam_module import MyVideoCapture
#import IR_cam_frame
#import RGB_cam_frame
#import simple_RGB_cam_frame
#from cam_module import MyVideoCapture


path = os.path.dirname(os.path.abspath(__file__))
#print('path: ', path)
#print('shared_variables.TEST', shared_variables.TEST)
#print(path+"/control_codes/process_detections.py")

#with open(path+"/control_codes/shared_csv_files/F2_log.csv","wb") as out, open(path+"/control_codes/shared_csv_files/F2_log_err.csv","wb") as err:

#subprocess.Popen(["python3", "control_codes/process_detections.py"], shell=True)


## Initialize 
UV0 = steer.UV()


def trunc(x):
    y = np.min([np.max([int(x), -100]), 100])
    return y

# Parameters




#////////////////////////////////////////////////////////////////////////////////////////////
#creating window 

root = tkinter.Tk()  
#root.geometry("780x630")

#root.attributes("-fullscreen", True)

#root.wm_title("Digital Microscope")

root.wm_attributes('-type','dock')

root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.focus_force()
root.wm_title("Digital Microscope")
#root.config(background='#BDBDBD')

#////////////////////////////////////////////////////////////////////////////////////////////

#Combobox


"""vlist = ["Option1", "Option2", "Option3","Option4"]
Combo = ttk.Combobox(root, values = vlist,font=" 6")
Combo.set("Pick an Option")
Combo.place(x=0,y=0)

#creat Start button
button =Button(root, text = "START",bd=5,activebackground='blue',width=55,height=1)
button.bind('<ButtonPress-1>', lambda event: start_button(Combo.get()))    
button.place(x=200,y=0)
 """

root.resizable(width=False, height=False)

password = StringVar()

test="maman1057"

def show(text):
    #p = password.get() #get password from entry
    if test==password.get():
        
        root.destroy()
    elif test!=password:
        text.delete('0',tkinter.END)
       
    
def close(app):
  app.destroy()



def password_set():
    #app=Toplevel(root)
    #app.geometry("500x300")
    
    #app=Toplevel(root)
    app=LabelFrame(root, text="password",height=200,width=200,bd=4)
    #app.geometry("200x100")
    #app.focus_force()
    #app.wm_title("Password")
    app.place(x=800,y=300)
    #app.geometry("100*100+300+300")
       
    #password = StringVar() #Password variable
    #passEntry = Entry(main, textvariable=password, show='*').pack() 
    text=ttk.Entry(app, width = 10, textvariable = password, show='*')
    text.pack()
    
    #text.bind('<Return>', show())
    submit = ttk.Button(app, text='Submit')
    submit.pack()
    submit.bind('<ButtonPress-1>', lambda event: show(text))
    
    submit = ttk.Button(app, text='Close')
    submit.pack()
    submit.bind('<ButtonPress-1>', lambda event: close(app))
    
    app.mainloop()
    
#root.protocol("WM_DELETE_WINDOW", password_set)

  

RB = tkinter.Button(root, text = "Close",width = 9,height=3,fg="red",bd=4,command=password_set)

RB.place(x=1405,y=0)
RB.bind('<ButtonPress-1>', lambda event: password_set)



#/////////////////////////////////////////////////////////////////////////////////////////////

#WIFI configuration

def setting():
    os.system("unity-control-center network")
  #print('WIFI Setting:')

R = tkinter.Button(root, text = "Connect to WIFI",width = 15,height=5,fg="blue",bd=4,command=setting)
R.place(x=1220,y=10)
R.bind('<ButtonPress-1>', lambda event: setting)




##############
def program_mode_set(mode):
    global program_mode
    program_mode = mode
    if mode == 'auto':
        subprocess.Popen(["python3 "+ path+"/control_codes/process_detections.py"],close_fds=True, shell=True) # stdout=out, stderr=err, 
        subprocess.Popen(["sudo python3 "+ path+"/control_codes/sanitize.py"],close_fds=True, shell=True) # stdout=out, stderr=err, 


   



var = IntVar()
R1 = Radiobutton(root, text="Manual", variable=var, value=1)
#R1.pack( anchor = W )
R1.place(x=5,y=3)

R1.bind('<Button-1>',lambda event: program_mode_set('manual'))


R2 = Radiobutton(root, text="Automatic", variable=var, value=2)
R2.place(x=110,y=3)
R2.bind('<Button-1>',lambda event: program_mode_set('auto'))

R3 = Radiobutton(root, text="Off", variable=var, value=3)
R3.select()
R3.place(x=230,y=3)
R3.bind('<Button-1>',lambda event: program_mode_set('off'))




#//////////////////////////////////////////////////////////////////////////////////////////////
#creating first LabelFrame container to add text box and buttons

labelframe_1 = LabelFrame(root, text="UV Engine 1",height=150,width=350,bd=4)

labelframe_1.place(x=0,y=25)
#labelframe_1.pack(side=LEFT)

#creating label of X and Y to add to the LabelFrame 
x_1 = Label(labelframe_1, text="X:")
x_1.place(x=0,y=20)
y_1=Label(labelframe_1, text="Y:")
y_1.place(x=140,y=20)

#creating text boxes to add to the LabelFrame 




vx1 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("x_unit1:",sv))

text1_unit1 = ttk.Entry(labelframe_1, width = 5, textvariable = vx1)

text1_unit1.place(x=15,y=20)
text1_unit1.insert(0,"0")



vy1 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("y_unit1:",sv))


text2_unit1 = ttk.Entry(labelframe_1, width = 5, textvariable = vy1)
text2_unit1.place(x=155,y=20)
text2_unit1.insert(0,"0")


def func1(event):
    print('UV1_X:',vx1.get())
    print('UV1_Y:',vy1.get())

    #
    #vx1.get() = np.min([np.max([vx1.get(), -300]), 300])
    # UV go
    if program_mode!='off':
        UV0.UV_go(LED=0,x=3*trunc(vx1.get()),y=3*trunc(vy1.get()),light_on = UV0.light_on[0])

    #print(text2_unit1.get())
text1_unit1.bind('<Return>', func1)
text2_unit1.bind('<Return>', func1)


#define function for buttons


Var1 = StringVar()


def default_color1():
    RB1["fg"]="#ff0000"
 
myCanvas1 = Canvas(labelframe_1,width=20, height=18)
myCanvas1.place(x=270,y=50)

Rbutton1=myCanvas1.create_oval(1, 3, 17, 18,fill="red", outline="#DDD")




x1=0
    
def turnon1():
    global x1
    if x1==0:
        
        #RB1["fg"]="#228B22"
        myCanvas1.itemconfig(Rbutton1, fill="green")
        x1=1
        print("Turn On")
        if program_mode!='off':
            UV0.UV_go(LED=0,x=3*trunc(vx1.get()),y=3*trunc(vy1.get()),light_on = True)

        
    elif x1==1:
        
        #RB1["fg"]="#ff0000"
        myCanvas1.itemconfig(Rbutton1, fill="red")
        x1=0
        print("Turn Off")
        if program_mode!='off':
            UV0.UV_go(LED=0,x=3*trunc(vx1.get()),y=3*trunc(vy1.get()),light_on = False)


    
    
       
    
#creating buttons to add to the LabelFrame    
button = tkinter.Button(labelframe_1, text = "Light On/Off",width = 9)
button.place(x=240,y=20)
button.bind('<ButtonPress-1>', lambda event: turnon1())








Rangex1=Label(labelframe_1, text="Range=(-100,100)")
Rangex1.place(x=0,y=50)
Rangey1=Label(labelframe_1, text="Range=(-100,100)")
Rangey1.place(x=130,y=50)


#///////////////////////////////////////////////////////////////////////////////////////////////

    
#///////////////////////////////////////////////////////////////////////////////////////////////
#UNIT_2

labelframe_2 = LabelFrame(root, text="UV Engine 2",height=150,width=350,bd=4)

labelframe_2.place(x=0,y=175)
#labelframe_2.pack(side=LEFT)

#creating label of X and Y to add to the LabelFrame 
x_1 = Label(labelframe_2, text="X:")
x_1.place(x=0,y=20)
y_1=Label(labelframe_2, text="Y:")
y_1.place(x=140,y=20)

#creating text boxes to add to the LabelFrame 

def print_text(string,value):
    print(string,value)

vx2 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("x_unit2:",vx2))

text1_unit2 = ttk.Entry(labelframe_2, width = 5, textvariable = vx2)
text1_unit2.place(x=15,y=20)
text1_unit2.insert(0,"0")


vy2 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("y_unit2:",vy2))


text2 = tkinter.IntVar()
text2_unit2 = ttk.Entry(labelframe_2, width = 5, textvariable = vy2)
text2_unit2.place(x=155,y=20)
text2_unit2.insert(0,"0")

def func2(event):
    print('UV2_X:',vx2.get())
    print('UV2_Y:',vy2.get())

    if program_mode!='off':
        UV0.UV_go(LED=1,x=3*trunc(vx2.get()),y=3*trunc(vy2.get()),light_on = UV0.light_on[1])

    #print(text2_unit1.get())
text1_unit2.bind('<Return>', func2)
text2_unit2.bind('<Return>', func2)

#buttons

"""button = ttk.Button(labelframe_2, text = "Go")
button.place(x=0,y=80)
button.bind('<ButtonPress-1>', lambda event: go(text1_unit1.get(),text2_unit1.get(),"unite_2"))

button = ttk.Button(labelframe_2, text = "Turn On")
button.place(x=100,y=80)
button.bind('<ButtonPress-1>', lambda event: turnon("unite_2"))"""


myCanvas2 = Canvas(labelframe_2,width=20, height=18)
myCanvas2.place(x=270,y=50)

Rbutton2=myCanvas2.create_oval(1, 3, 17, 18,fill="red", outline="#DDD")



x2=0
    
def turnon2():
    global x2
    if x2==0:
        
        #RB2["fg"]="#228B22"
        myCanvas2.itemconfig(Rbutton2, fill="green")
        x2=1
        print("Turn On")
        if program_mode!='off':
            UV0.UV_go(LED=1,x=3*trunc(vx2.get()),y=3*trunc(vy2.get()),light_on = True)
        
    elif x2==1:
        
        #RB2["fg"]="#ff0000"
        myCanvas2.itemconfig(Rbutton2, fill="red")
        x2=0
        print("Turn Off")
        if program_mode!='off':
            UV0.UV_go(LED=1,x=3*trunc(vx2.get()),y=3*trunc(vy2.get()),light_on = False)

        
       
    
#creating buttons to add to the LabelFrame    
button = tkinter.Button(labelframe_2, text = "Light On/Off",width = 9)
button.place(x=240,y=20)
button.bind('<ButtonPress-1>', lambda event: turnon2())



Rangex2=Label(labelframe_2, text="Range=(-100,100)")
Rangex2.place(x=0,y=50)
Rangey2=Label(labelframe_2, text="Range=(-100,100)")
Rangey2.place(x=130,y=50)


#//////////////////////////////////////////////////////////////////////////////////////
#UNIT_3
#creat LabelFrame
labelframe_3 = LabelFrame(root, text="UV Engine 3",height=150,width=350,bd=4)
labelframe_3.place(x=0,y=325)
#labelframe_3.pack(side=LEFT)

#labels

x_3 = Label(labelframe_3, text="X:")
x_3.place(x=0,y=20)
y_2=Label(labelframe_3, text="Y:")
y_2.place(x=140,y=20)

#textboxes

def print_text(string,value):
    print(string,value)

vx3 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("x_unit3:",sv))

text1_unit3 = ttk.Entry(labelframe_3, width = 5, textvariable = vx3)
text1_unit3.place(x=15,y=20)
text1_unit3.insert(0,"0")

vy3 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("y_unit3:",sv))

text2_unit3 = ttk.Entry(labelframe_3, width = 5, textvariable = vy3)
text2_unit3.place(x=155,y=20)
text2_unit3.insert(0,"0")


def func3(event):
    print('UV3_X:',vx3.get())
    print('UV3_Y:',vy3.get())

    if program_mode!='off':
        UV0.UV_go(LED=2,x=3*trunc(vx3.get()),y=3*trunc(vy3.get()),light_on = UV0.light_on[2])

    #print(text2_unit1.get())
text1_unit3.bind('<Return>', func3)
text2_unit3.bind('<Return>', func3)



#buttons
"""
button = ttk.Button(labelframe_3, text = "Go")
button.place(x=0,y=80)
button.bind('<ButtonPress-1>', lambda event: go(text1_unit1.get(),text2_unit1.get(),"unite_3"))

button = ttk.Button(labelframe_3, text = "Turn On")
button.place(x=100,y=80)
button.bind('<ButtonPress-1>', lambda event: turnon("unite_3"))"""


myCanvas3 = Canvas(labelframe_3,width=20, height=18)
myCanvas3.place(x=270,y=50)

Rbutton3=myCanvas3.create_oval(1, 3, 17, 18,fill="red", outline="#DDD")



x3=0
    
def turnon3():
    global x3
    if x3==0:
        
        #RB3["fg"]="#228B22"
        myCanvas3.itemconfig(Rbutton3, fill="green")
        x3=1
        print("Turn On")
        if program_mode!='off':
            UV0.UV_go(LED=2,x=3*trunc(vx3.get()),y=3*trunc(vy3.get()),light_on = True)
        
    elif x3==1:
        
        #RB3["fg"]="#ff0000"
        myCanvas3.itemconfig(Rbutton3, fill="red")

        x3=0
        print("Turn Off")
        if program_mode!='off':
            UV0.UV_go(LED=2,x=3*trunc(vx3.get()),y=3*trunc(vy3.get()),light_on = False)

        
       
    
#creating buttons to add to the LabelFrame    
button = tkinter.Button(labelframe_3, text = "Light On/Off",width = 9)
button.place(x=240,y=20)
button.bind('<ButtonPress-1>', lambda event: turnon3())





Rangex3=Label(labelframe_3, text="Range=(-100,100)")
Rangex3.place(x=0,y=50)
Rangey3=Label(labelframe_3, text="Range=(-100,100)")
Rangey3.place(x=130,y=50)


#///////////////////////////////////////////////////////////////////////////////
#UNIT_4
#creating LabelFrame

labelframe_4 = LabelFrame(root, text="UV Engine 4",height=150,width=350,bd=4)
labelframe_4.place(x=0,y=475)
#labelframe_4.pack(side=LEFT)

#labels
x_4 = Label(labelframe_4, text="X:")
x_4.place(x=0,y=20)
y_4=Label(labelframe_4, text="Y:")
y_4.place(x=140,y=20)


#textboxes
def print_text(string,value):
    print(string,value)

vx4 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("x_unit4:",sv))


text1_unit4 = ttk.Entry(labelframe_4, width = 5, textvariable = vx4)
text1_unit4.place(x=15,y=20)
text1_unit4.insert(0,"0")


vy4 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("y_unit4:",sv))
text2_unit4 = ttk.Entry(labelframe_4, width = 5, textvariable = vy4)
text2_unit4.place(x=155,y=20)
text2_unit4.insert(0,"0")



def func4(event):
    print('UV4_X:',vx4.get())
    print('UV4_Y:',vy4.get())

    if program_mode!='off':
        UV0.UV_go(LED=3,x=3*trunc(vx4.get()),y=3*trunc(vy4.get()),light_on = UV0.light_on[3])

    #print(text2_unit1.get())
text1_unit4.bind('<Return>', func4)
text2_unit4.bind('<Return>', func4)


#buttons
"""
button = ttk.Button(labelframe_4, text = "Go")
button.place(x=0,y=80)
button.bind('<ButtonPress-1>', lambda event: go(text1_unit1.get(),text2_unit1.get(),"unite_4"))



button = ttk.Button(labelframe_4, text = "Turn On")
button.place(x=100,y=80)
button.bind('<ButtonPress-1>', lambda event: turnon("unite_4"))"""


myCanvas4 = Canvas(labelframe_4,width=20, height=18)
myCanvas4.place(x=270,y=50)

Rbutton4=myCanvas4.create_oval(1, 3, 17, 18,fill="red", outline="#DDD")


x4=0
    
def turnon4():
    global x4
    if x4==0:
        
        #RB4["fg"]="#228B22"
        myCanvas4.itemconfig(Rbutton4, fill="green")

        x4=1
        print("Turn On")
        if program_mode!='off':
            UV0.UV_go(LED=3,x=3*trunc(vx4.get()),y=3*trunc(vy4.get()),light_on = True)
        
    elif x4==1:
        
        #RB4["fg"]="#ff0000"
        myCanvas4.itemconfig(Rbutton4, fill="red")
        x4=0
        print("Turn Off")
        if program_mode!='off':
            UV0.UV_go(LED=3,x=3*trunc(vx4.get()),y=3*trunc(vy4.get()),light_on = False)

        
 
    
       
    
#creating buttons to add to the LabelFrame    
button = tkinter.Button(labelframe_4, text = "Light On/Off",width = 9)
button.place(x=240,y=20)
button.bind('<ButtonPress-1>', lambda event: turnon4())



Rangex4=Label(labelframe_4, text="Range=(-100,100)")
Rangex4.place(x=0,y=50)
Rangey4=Label(labelframe_4, text="Range=(-100,100)")
Rangey4.place(x=130,y=50)
#/////////////////////////////////////////////////////////////////////////////////////

label_1 = Label(root, text="RGB Camera",height=5,width=10,bd=4,font="bold")
label_1.place(x=360,y=140)



label_2 = Label(root, text="IR Camera",height=5,width=10,bd=4,font="bold")
label_2.place(x=360,y=430)


# ////////////////////////////

 
#//////////////////////////////////////////////////////////////////////////////////
#creating video fram

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        #self.video_source = video_source
        
        # open video source
        self.vid = MyVideoCapture(sensor_id=0)
        
        #self.video_source_2 = video
        self.vid_2 = MyIRCapture(sensor_id=1)
        
        # Create a canvas that can fit the above video source size
        
        self.canvas_1 = tkinter.Canvas(window, width = 640, height = 480)
        self.canvas_1.place(x=480,y=0)
                
        self.canvas_2 = tkinter.Canvas(window, width = 640, height = 480)
        self.canvas_2.place(x=480,y=490)
        
        self.delay = 15
        self.update()
        
        self.window.mainloop()
        
    def update(self):
        
        # Select the program mode
        if program_mode=='auto':
            ret, frame = self.vid.get_processed_frame()#frame()#get_processed_frame()
            ret_2,frame_2=self.vid_2.get_processed_IR_frame()#frame()#get_processed_frame()
        
        else:
            ret, frame = self.vid.get_frame()
            ret_2,frame_2 = self.vid_2.get_frame() #self.vid_2.get_frame()

        #print(frame.shape[0])

        DISPLAY_WIDTH=640
        DISPLAY_HEIGHT=480
        #frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        #frame_2 = cv2.resize(frame_2, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        #ret,frame=self.vid.get_frame()
        #ret_2,frame_2=self.vid_2.get_frame()

        if ret :
            
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            #self.photo_2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_2))
            self.canvas_1.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            #self.canvas_2.create_image(0, 0, image = self.photo_2, anchor = tkinter.NW)
            
        if ret_2:
            self.photo_2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_2))
            self.canvas_2.create_image(0, 0, image = self.photo_2, anchor = tkinter.NW)
            
        self.window.after(self.delay, self.update)
        
        

App(root, "Shyld AI") 


#cv2image = my_cam_obj.get_frame()
#os.popen("bash lock_screen.txt")
#subprocess.Popen(["bash", "lock_screen.txt"])
root.mainloop()





