import time
import cv2
import numpy as np
import os
from datetime import datetime


# sensor_id=1 ... left camera
# sensor_id=0 ... right camera
sensor_id = 1
assert(sensor_id == 0 or sensor_id == 1)


# File for captured image
if sensor_id == 1:
    file_path = './img_left/'
else:
    file_path = './img_right/'
file_prefix = 'img_'
file_suffix = '.png'

# Displayed image size
scale_factor = 0.5

# Controls
print("Controls:")
print("s ... Grab Frame")
print("q ... Quit")

# Camera settinge 
# 1920x1080
cam_width = 1920
cam_height = 1080
print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Displayed image size
img_width = int (cam_width * scale_factor)
img_height = int (cam_height * scale_factor)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Gstreamer pipeline for rpi-hq-cam (with arguscam driver) 
cap_receive = cv2.VideoCapture('nvarguscamerasrc sensor-id=' + str(sensor_id) + ' ! video/x-raw(memory:NVMM), width=(int)' + str(cam_width) +', height=(int)' + str(cam_height) + ', format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! nvvidconv ! appsink', cv2.CAP_GSTREAMER)


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
   
    
