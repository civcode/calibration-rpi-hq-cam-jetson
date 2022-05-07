import cv2
import os
import numpy as np
import json
import pickle
import glob


# sensor_id=1 ... left camera
# sensor_id=0 ... right camera
sensor_id = 0
assert(sensor_id == 0 or sensor_id == 1)

# Sensor parameters
# imx477
pixel_size = 1.55e-6 # in m, square
aperture_size = [pixel_size*4056, pixel_size*3040] # active sensor area

# File for captured image
if sensor_id == 1:
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


dim = calib_data["dim"]
K = calib_data["K"]
print(dim)
print(K)

fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(K, dim, aperture_size[0], aperture_size[1])

print('\ndim: ', dim)
print('fovx: ', fovx)
print('fovy: ', fovy)
print('focal length: ', focalLength)
print('principal point: ', principalPoint)
print('aspect ratio: ', aspectRatio)



