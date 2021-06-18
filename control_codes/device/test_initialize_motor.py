from steer import UV
import os
import pandas as pd
import time 
from UV_coordinate_initialize import init_coordinates





#write_light_loc_from_IR(150,150)
my_UV = UV()

my_UV.init_loc(LED=0)
print('initialized')
c = my_UV.read_IR_locations()
print(c)
my_UV.reset_loc(LED=0)
print('loc reset')