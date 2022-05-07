
import cv2
import os
#from picamera.array import PiRGBArray
#from picamera import PiCamera
#from matplotlib import pyplot as plt
#from matplotlib.widgets import Slider, Button
import numpy as np
import json
import pickle
import glob



cam_num = 0
assert(cam_num == 0 or cam_num == 1)

if cam_num == 0:
    file_path = './img_left/'
    data_name = '/calib_left.dat'
else:
    file_path = './img_right/'
    data_name = '/calib_right.dat'
    
    
data_path = os.getcwd() + data_name
print(data_path)

with open(data_path, "rb") as f:
    calib_data = pickle.load(f)
    #print(pickle.load(f))

#pixel_size = 1.4e-6 #m, square
#image_area = (3673.6e-6/2, 2738.4e-6/2) #chip size in m 
pixel_size = 1.75e-6 #m, square
aperture_size = [pixel_size*1280e-6, pixel_size*480e-6]

dim = calib_data["dim"]
K = calib_data["K"]
print(dim)
print(K)

#dim = (1280, 480)
fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(
    K, dim, aperture_size[0], aperture_size[1])

print('dim: ', dim)
print('fovx: ', fovx)
print('fovy: ', fovy)
print('focal length: ', focalLength)
print('principal point: ', principalPoint)
print('aspect ratio: ', aspectRatio)



