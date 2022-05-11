import time
import cv2
import numpy as np
import os


# sensor_id=1 ... left camera
# sensor_id=0 ... right camera
sensor_id = 1
assert(sensor_id == 0 or sensor_id == 1)

# Displayed image size
scale_factor = 0.5

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


print("\nq ... Quit")
print("s ... Select ROI\n")

counter = 0
avgtime = 0
file_idx = 0
get_roi = False 
roi = []

while True:
    ret, frame = cap_receive.read()
    
    img = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    if get_roi:
        roi = cv2.selectROI(img)
        roi = [int(val/scale_factor) for val in roi]
        print(roi)
        get_roi = False
        cv2.destroyWindow("ROI selector")

    if not roi:
        cv2.imshow("roi", img)
    else:
        img_cropped = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        img_cropped = cv2.resize(img_cropped, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("roi", img_cropped)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        get_roi = True
    
    if key == ord("q"):
        cv2.destroyAllWindows()
        exit()
   
    
