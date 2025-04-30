#!/usr/bin/python3

import cv2
import numpy as np
import os
import glob
import pickle
import sys

import config

sensor_id = config.sensor_id

# File for captured image
data_name = config.data_name
file_path = config.file_path

if not os.path.exists(file_path):
    print("\nPath " + str(file_path) + " does not exist.")
    print("No images found.")
    quit()

# Displayed image size
scale_factor = config.scale_factor

# Calibration pattern parameters
CHECKERBOARD = (6,9)
square_size = 19.8e-2 / 8

corner_subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1E-5)

calib_flags = (cv2.CALIB_USE_INTRINSIC_GUESS)

calib_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)


objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float32)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(file_path + '/*' + config.file_suffix)
images.sort()

if not images:
    print('No images found in "{}"'.format(img_path_expr))
    quit()

for fname in images:
    img = cv2.imread(fname)
    w,h = img.shape[:2]
    img_width = int(w*scale_factor)
    img_heigth = int(h*scale_factor)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    rms, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if rms == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners, (3,3), (-1,-1), corner_subpix_criteria)
        imgpoints.append(corners)
        
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, rms)
        #cv2.imshow('img', img)
        img_scaled = cv2.resize(img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('img', img_scaled)
        cv2.waitKey(500)
        
        
N_OK = len(objpoints)

img_size = (w, h)
rms, camera_matrix, dist_coeffs, rvecs, tvecs = \
    cv2.calibrateCamera(
        objpoints, 
        imgpoints, 
        img_size, 
        None,
        None,
        # flags=calib_flags, 
        criteria=calib_criteria
    )

new_K, valid_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (h, w), 1, centerPrincipalPoint=False)

calib_data = {"dim": _img_shape[::-1], "K": new_K, "D": dist_coeffs, "type": "normal"}
pickle_file = os.getcwd() + data_name
with open(pickle_file, "wb") as f:
    pickle.dump(calib_data, f)

# print('Image size:', gray.shape[::-1])
# print("rms=" + str(rms))
# print("Camera matirx=")
# print(camera_matrix)
# print("Distortion parameters=")
# print(dist_coeffs)
# print("Valid roi:")
# print(valid_roi)

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("rms=" + str(rms))
print("K=np.array(" + str(new_K.tolist()) + ")")
print("D=np.array(" + str(dist_coeffs.tolist()) + ")")

for fname in images:
    img = cv2.imread(fname)
    w,h = img.shape[:2]
    #undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_K)
    #cv2.imshow('img', img)
    #cv2.imshow("undistorted", undistorted_img)
    img_width = int(w*scale_factor)
    img_heigth = int(h*scale_factor)
    img_scaled = cv2.resize(img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
    undistorted_scaled = cv2.resize(undistorted_img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('img', img_scaled)
    cv2.imshow("undistorted", undistorted_scaled)
    #print('image')
    key = cv2.waitKey(0)
    if key == ord('q'):
        quit()

 
cv2.destroyAllWindows()

