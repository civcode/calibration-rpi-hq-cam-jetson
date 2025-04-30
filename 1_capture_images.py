#!/usr/bin/python3

import time
import cv2
import numpy as np
import os
from datetime import datetime

import config

sensor_id = config.sensor_id

# File for captured image
data_name = config.data_name
file_path = config.file_path
file_prefix = config.file_prefix
file_suffix = config.file_suffix

# Displayed image size
scale_factor = config.scale_factor

# Controls
print("Controls:")
print("s ... Grab Frame")
print("q ... Quit")

# Camera settinge 
cam_width = 640
cam_height = 480
print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Displayed image size
img_width = int (cam_width * scale_factor)
img_height = int (cam_height * scale_factor)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# PS3 Eye v4l2 driver
cap_receive = cv2.VideoCapture(sensor_id, cv2.CAP_V4L2)


if not cap_receive.isOpened():
    print('VideoCapture not opened')
    quit()


t2= datetime.now()

counter = 0
avgtime = 0
file_idx = 0

# Capture frames from the camera
while True:
    ret, frame = cap_receive.read()
    
    counter+=1
    t1 = datetime.now()
    timediff = t1-t2
    avgtime = avgtime + (timediff.total_seconds())
    img = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("cam", frame)
    cv2.imshow("cam", img)
    key = cv2.waitKey(1) & 0xFF
    t2 = datetime.now()
    
    if key == ord("q") :
        avgtime = avgtime/counter
        print ("Average time between frames: " + str(avgtime))
        print ("Average FPS: " + str(1/avgtime))
        cv2.destroyAllWindows()
        exit()
    elif key == ord("s"):
        if (os.path.isdir(file_path)==False):
            os.makedirs(file_path)    
            
        idx_string = "{:03}".format(file_idx)
        cv2.imwrite(file_path+file_prefix+idx_string+file_suffix, frame)
        file_idx += 1
        print("captured image " + str(file_idx))
   
    
