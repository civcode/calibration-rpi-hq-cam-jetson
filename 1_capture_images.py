
#import picamera
#from picamera import PiCamera
import time
import cv2
import numpy as np
import os
from datetime import datetime


# 0: left cam, 1: right cam
# File for captured image
cam_num = 0
assert(cam_num == 0 or cam_num == 1)


filename = './img/photo.png'
if cam_num == 0:
    file_path = './img_left/'
else:
    file_path = './img_right/'
file_prefix = 'img_'
file_suffix = '.png'

#file_path = './img_test/'

# Camera settimgs
cam_width = 640 
cam_height = 480 
fps = 30

# Final image capture settings
scale_ratio = 0.5

# controls
print("Controls:")
print("s ... Grab Frame")
print("q ... Quit")

# Camera resolution 
# 1920x1080
cam_width = 1920
cam_height = 1080
print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# displayed image size
img_width = int (cam_width * scale_ratio)
img_height = int (cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Initialize the camera
#camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
#camera = PiCamera(camera_num=cam_num, led_pin=1)
#camera.resolution=(cam_width, cam_height)
#camera.framerate = fps
#camera.hflip = False #True

# capture pipeline for orlaco camera
#cap_receive = cv2.VideoCapture('udpsrc multicast-group=239.255.255.200 multicast-iface=eth0 auto-multicast=true port=50008 ! application/x-rtp, encoding-name=JPEG, payload=26 ! rtpjpegdepay ! vaapijpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

# pipeline for rpi-hq-cam 
#cap_receive = cv2.VideoCapture('nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! nvvidconv ! appsink', cv2.CAP_GSTREAMER)
cap_receive = cv2.VideoCapture('nvarguscamerasrc sensor-id=' + str(cam_num) + ' ! video/x-raw(memory:NVMM), width=(int)' + str(cam_width) +', height=(int)' + str(cam_height) + ', format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! nvvidconv ! appsink', cv2.CAP_GSTREAMER)


if not cap_receive.isOpened():
    print('VideoCapture not opened')
    quit()


t2 = datetime.now()
counter = 0
avgtime = 0
file_idx = 0
# Capture frames from the camera
#for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
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
    # if the `q` key was pressed, break from the loop and save last image
    if key == ord("q") :
        avgtime = avgtime/counter
        print ("Average time between frames: " + str(avgtime))
        print ("Average FPS: " + str(1/avgtime))
        #if (os.path.isdir("./scenes")==False):
        #    os.makedirs("./scenes")
        ##if (os.path.isdir(file_path)==False):
        ##    os.makedirs(file_path)    
            
        ##cv2.imwrite(filename, frame)
        cv2.destroyAllWindows()
        exit()
    elif key == ord("s"):
        if (os.path.isdir(file_path)==False):
            os.makedirs(file_path)    
            
        idx_string = "{:03}".format(file_idx)
        cv2.imwrite(file_path+file_prefix+idx_string+file_suffix, frame)
        print("captured image " + str(file_idx))
        file_idx += 1
   
    