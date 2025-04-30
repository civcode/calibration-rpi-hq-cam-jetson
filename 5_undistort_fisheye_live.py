#!/usr/bin/python3

import cv2
import glob
import numpy as np
import os
import pickle


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

# Camera settinge 
cam_width = 640
cam_height = 480
print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Displayed image size
img_width = int (cam_width * scale_factor)
img_height = int (cam_height * scale_factor)
#capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

    
data_path = os.getcwd() + data_name
print(data_path)

with open(data_path, "rb") as f:
    calib_data = pickle.load(f)
    #print(pickle.load(f))

for item in calib_data:
    print(item)

print('dim =', calib_data["dim"])
print('K =', calib_data["K"])
print('D =', calib_data["D"])


# test calibratin results
#DIM = _img_shape[::-1]
#dim1 = DIM
dim = calib_data["dim"]
K = calib_data["K"]
D = calib_data["D"]
type = calib_data["type"]

if type != 'fisheye':
    print("\nCalibration data type is not fisheye")
    print("Run mathing calibration")
    quit()

balance = 0.0
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)


cap_receive = cv2.VideoCapture(sensor_id, cv2.CAP_V4L2)

if not cap_receive.isOpened():
    print('VideoCapture not opened')
    quit()

#cv2.namedWindow('img', cv2.WINDOW_NORMAL)
#cv2.namedWindow('undistorted', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('undistorted', 1280, 960)
#cv2.namedWindow('perspective', cv2.WINDOW_NORMAL)
# Capture frames from the camera
#for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
while True:
    ret, frame = cap_receive.read()

    #print(img.shape)
    h,w = frame.shape[:2]
    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    img = cv2.resize(undistorted_img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('img', img)
    #cv2.imshow("undistorted", undistorted_img)
    #print('image')
    key = cv2.waitKey(1)
    if key == ord("q"):
        quit()
    
    if key == ord("j") or key == ord("k"):
        if key == ord("k"):
            if balance < 0.9:
                balance += 0.1
                print('balance: {0:.2f}'.format(balance))
        elif key == ord("j"):
            if balance > 0.1:
                balance -= 0.1
                print('balance: {0:.2f}'.format(balance))
                
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
        #undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", img)
    #key = cv2.waitKey(1)

    # src_points = np.float32( [ [0,800], [500,400], [1280-500,400], [1280,800] ] )
    # dst_points = np.float32( [ [0,960], [0,0], [1280,0], [1280,960] ] )

    # img, out = perspective_transform(undistorted_img, src_points, dst_points )
    # cv2.imshow("perspective", out)
    #key = cv2.waitKey(1)
    
    
 
cv2.destroyAllWindows()
