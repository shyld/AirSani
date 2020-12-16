# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:01:04 2020

@author: Administrator
"""

import tkinter

from tkinter import *

from tkinter import ttk

import PIL

from PIL import Image, ImageTk

import datetime
import cv2
import os

# Parameters




#////////////////////////////////////////////////////////////////////////////////////////////
#creating window 

root = tkinter.Tk()  
root.geometry("780x630")

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


def sel(st):
   print(st)
   



var = IntVar()
R1 = Radiobutton(root, text="Manual", variable=var, value=1)
#R1.pack( anchor = W )
R1.place(x=5,y=3)

R1.bind('<Button-1>',lambda event: sel(R1["text"]))


R2 = Radiobutton(root, text="Automatic", variable=var, value=2)
R2.place(x=110,y=3)
R2.bind('<Button-1>',lambda event: sel(R2["text"]))

R3 = Radiobutton(root, text="Off", variable=var, value=3)

R3.place(x=230,y=3)
R3.bind('<Button-1>',lambda event: sel(R3["text"]))




#//////////////////////////////////////////////////////////////////////////////////////////////
#creating first LabelFrame container to add text box and buttons

labelframe_1 = LabelFrame(root, text="UV Unit_1",height=150,width=350,bd=4)
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




vy1 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("y_unit1:",sv))


text2_unit1 = ttk.Entry(labelframe_1, width = 5, textvariable = vy1)
text2_unit1.place(x=155,y=20)


def func1(event):
    print('UV1_X:',vx1.get())
    print('UV1_Y:',vy1.get())
    #print(text2_unit1.get())
text1_unit1.bind('<Return>', func1)
text2_unit1.bind('<Return>', func1)


#define function for buttons


Var1 = StringVar()


def default_color1():
    RB1["fg"]="#ff0000"
 
RB1 =tkinter.Radiobutton(labelframe_1, variable = Var1,value = 1,fg="#ff0000",command = default_color1)
RB1.place(x=270,y=50)




x1=0
    
def turnon1():
    global x1
    if x1==0:
        
        RB1["fg"]="#228B22"
        x1=1
        print("Turn On")
        
    elif x1==1:
        
        RB1["fg"]="#ff0000"
        x1=0
        print("Turn Off")



    
    
       
    
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

labelframe_2 = LabelFrame(root, text="UV Unit_2",height=150,width=350,bd=4)
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


vy2 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("y_unit2:",vy2))


text2 = tkinter.IntVar()
text2_unit2 = ttk.Entry(labelframe_2, width = 5, textvariable = vy2)
text2_unit2.place(x=155,y=20)

def func2(event):
    print('UV2_X:',vx2.get())
    print('UV2_Y:',vy2.get())
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


Var2 = StringVar()

def default_color2():
    RB2["fg"]="#ff0000"
 
RB2 = Radiobutton(labelframe_2, variable = Var1,value = 1,fg="#ff0000",command = default_color2)


RB2.place(x=270,y=50)


x2=0
    
def turnon2():
    global x2
    if x2==0:
        
        RB2["fg"]="#228B22"
        x2=1
        print("Turn On")
        
    elif x2==1:
        
        RB2["fg"]="#ff0000"
        x2=0
        print("Turn Off")

        
       
    
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
labelframe_3 = LabelFrame(root, text="UV Unit_3",height=150,width=350,bd=4)
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

vy3 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("y_unit3:",sv))

text2_unit3 = ttk.Entry(labelframe_3, width = 5, textvariable = vy3)
text2_unit3.place(x=155,y=20)


def func3(event):
    print('UV3_X:',vx3.get())
    print('UV3_Y:',vy3.get())
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


Var3 = StringVar()

def default_color3():
    RB3["fg"]="#ff0000"
 
RB3 = Radiobutton(labelframe_3, variable = Var1,value = 1,fg="#ff0000",command = default_color3)
RB3.place(x=270,y=50)



x3=0
    
def turnon3():
    global x3
    if x3==0:
        
        RB3["fg"]="#228B22"
        x3=1
        print("Turn On")
        
    elif x3==1:
        
        RB3["fg"]="#ff0000"
        x3=0
        print("Turn Off")

        
       
    
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

labelframe_4 = LabelFrame(root, text="UV Unit_4",height=150,width=350,bd=4)
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


vy4 = StringVar()
#sv.trace("w", lambda name, index, mode, sv=sv: callback("y_unit4:",sv))
text2_unit4 = ttk.Entry(labelframe_4, width = 5, textvariable = vy4)
text2_unit4.place(x=155,y=20)



def func4(event):
    print('UV4_X:',vx4.get())
    print('UV4_Y:',vy4.get())
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


Var4 = StringVar()

def default_color4():
    RB4["fg"]="#ff0000"
 
RB4 = Radiobutton(labelframe_4, variable = Var1,value = 1,fg="#ff0000",command = default_color4)
RB4.place(x=270,y=50)



x4=0
    
def turnon4():
    global x4
    if x4==0:
        
        RB4["fg"]="#228B22"
        x4=1
        print("Turn On")
        
    elif x4==1:
        
        RB4["fg"]="#ff0000"
        x4=0
        print("Turn Off")

        
 
    
       
    
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




 
#//////////////////////////////////////////////////////////////////////////////////
#creating video fram

class App:
    def __init__(self, window, window_title, video_source,video,X,Y):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        
        # open video source
        self.vid = MyVideoCapture(video_source)
        
        self.video_source_2 = video
        self.vid_2=MyVideoCapture(video)
        
        # Create a canvas that can fit the above video source size
        
        self.canvas_1 = tkinter.Canvas(window, width = 300, height = 300)
        self.canvas_1.place(x=X,y=Y)
                
        self.canvas_2 = tkinter.Canvas(window, width = 300, height = 312)
        self.canvas_2.place(x=470,y=300)
        
        self.delay = 15
        self.update()
        
        self.window.mainloop()
        
    def update(self):
        
        ret, frame = self.vid.get_frame()
        ret_2,frame_2=self.vid_2.get_frame()
        if ret :
            
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            #self.photo_2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_2))
            self.canvas_1.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            #self.canvas_2.create_image(0, 0, image = self.photo_2, anchor = tkinter.NW)
            
        if ret_2:
            self.photo_2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_2))
            self.canvas_2.create_image(0, 0, image = self.photo_2, anchor = tkinter.NW)
            
        self.window.after(self.delay, self.update)
        
        
# Create a window and pass it to the Application object
   

class MyVideoCapture:
    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
            
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.mainloop()
        
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
 
video_source='01.mp4'

video='02.mp4'

App(root, "Tkinter and OpenCV",video_source,video,470,33) 



root.mainloop()


#/////////////////////////////////////////////////////////////////////////////////////////////////////







