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

if not os.path.exists(data_path):
    print("\nPath des not exist.")
    print(data_path)
    sys.exit()

with open(data_path, "rb") as f:
    calib_data = pickle.load(f)
    #print(pickle.load(f))


dim = calib_data["dim"]
K = calib_data["K"]
print(dim)
print(K)

fovv, fovh, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(K, dim, aperture_size[0], aperture_size[1])

print('\ndim: ', dim)
print('fovv: ', fovv)
print('fovh: ', fovh)
print('focal length: ', focalLength)
print('principal point: ', principalPoint)
print('aspect ratio: ', aspectRatio)



