# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:36:09 2020

@author: Administrator
"""



import tkinter

from tkinter import *

from tkinter import ttk
import time
import os
import datetime
import pandas as pd

from subprocess import Popen
import sys


Popen(["unity-control-center network"], shell=True)

os.system("xrandr --fb 1500x1000")

# report process running
df = pd.DataFrame({'process':'boot-run','time':[datetime.datetime.now()]})
df.to_csv('/home/shyldai/shyld/log_GUI.csv')


root = tkinter.Tk() 


root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.resizable(width=FALSE, height=FALSE)


root.wm_title("Digital Microscope")

root.after(20000, lambda: root.destroy()) 
#root.config(background='#BDBDBD')

#////////////////////////////////////////////////////////////////////////////////////////////




root.resizable(width=False, height=False)



x_1 = Label(root, text="setting up ...")
x_1.config(font=("Courier", 24))
x_1.place(x=620,y=380)




#time.sleep(30)

#root.destroy()

root.mainloop()


