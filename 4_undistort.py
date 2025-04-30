#!/usr/bin/python3

import cv2
import glob
import numpy as np
import os
import pickle

import config

sensor_id = config.sensor_id

data_name = config.data_name
file_path = config.file_path

if not os.path.exists(file_path):
    print("\nPath " + str(file_path) + " does not exist.")
    print("No images found.")
    quit()

# Displayed image size
scale_factor = config.scale_factor

    
data_path = os.getcwd() + data_name
print(data_path)

with open(data_path, "rb") as f:
    calib_data = pickle.load(f)

print('dim =', calib_data["dim"])
print('K =', calib_data["K"])
print('D =', calib_data["D"])

dim = calib_data["dim"]
K = calib_data["K"]
D = calib_data["D"]
type = calib_data["type"]

if type != 'normal':
    print("\nCalibration data type is not normal")
    print("Run matching calibration")
    quit()

balance = 0.0
new_K, valid_roi = cv2.getOptimalNewCameraMatrix(K, D, dim, alpha=balance, centerPrincipalPoint=False)
map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, dim, cv2.CV_16SC2)

print("\nPress j to decrease balance value")
print("Press k to increase balance value")

images = glob.glob(file_path + '/*' + config.file_suffix)
if len(images) == 0:
    print("\nNo images found in ", file_path)

for fname in images:
    img = cv2.imread(fname)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    w,h = img.shape[:2]
    img_width = int(w*scale_factor)
    img_heigth = int(h*scale_factor)
    
    img_scaled = cv2.resize(img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
    undistorted_scaled = cv2.resize(undistorted_img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
    
    cv2.imshow('img', img_scaled)
    cv2.imshow("undistorted", undistorted_scaled)
    
    key = cv2.waitKey(0)
    if key == ord("q"):
        quit()
    while key == ord("j") or key == ord("k"):
        increment = 0.1
        if key == ord("k"):
            if balance <= 1.0-increment:
                balance += increment 
                print('balance: {0:.2f}'.format(balance))
        elif key == ord("j"):
            if balance >= 0.0+increment:
                balance -= increment
                print('balance: {0:.2f}'.format(balance))
                
        new_K, valid_roi = cv2.getOptimalNewCameraMatrix(K, D, dim, alpha=balance, centerPrincipalPoint=False)
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, dim, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        undistorted_scaled = cv2.resize(undistorted_img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("undistorted", undistorted_scaled)
        key = cv2.waitKey(0)
        if key == ord("q"):
            quit()
        
cv2.destroyAllWindows()
