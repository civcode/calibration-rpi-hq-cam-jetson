import cv2
import glob
import numpy as np
import os
import pickle

def perspective_transform(img, src_points, dst_points, inverse=False):
    if inverse:
        src = dst_points
        dst = src_points
    else:
        src = src_points
        dst = dst_points
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = img.shape[1::-1]
    out = img.copy()
    cv2.polylines( out, [src.astype(np.int32)], isClosed=False, color=(255,0,0), thickness=1 )
    #warped = cv2.warpPerspective(out, M, img_size, flags=cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(out, M, img_size, flags=cv2.INTER_CUBIC)
    
    return out, warped

cam_num = 0
assert(cam_num == 0 or cam_num == 1)

# Camera settinge 
# 1920x1080
cam_width = 1920
cam_height = 1080
print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))


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

balance = 0.5
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)

#images = glob.glob(file_path + '/*.png')

# capture pipeline for orlaco camera
#cap_receive = cv2.VideoCapture('udpsrc multicast-group=239.255.255.200 multicast-iface=eth0 auto-multicast=true port=50008 ! application/x-rtp, encoding-name=JPEG, payload=26 ! rtpjpegdepay ! vaapijpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
cap_receive = cv2.VideoCapture('nvarguscamerasrc sensor-id=' + str(cam_num) + ' ! video/x-raw(memory:NVMM), width=(int)' + str(cam_width) +', height=(int)' + str(cam_height) + ', format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! nvvidconv ! appsink', cv2.CAP_GSTREAMER)

if not cap_receive.isOpened():
    print('VideoCapture not opened')
    quit()

#cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.namedWindow('undistorted', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('undistorted', 1280, 960)
#cv2.namedWindow('perspective', cv2.WINDOW_NORMAL)
# Capture frames from the camera
#for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
while True:
    ret, img = cap_receive.read()

    #print(img.shape)
    h,w = img.shape[:2]
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #cv2.imshow('img', img)
    #cv2.imshow("undistorted", undistorted_img)
    #print('image')
    key = cv2.waitKey(0)
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
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    #key = cv2.waitKey(1)

    # src_points = np.float32( [ [0,800], [500,400], [1280-500,400], [1280,800] ] )
    # dst_points = np.float32( [ [0,960], [0,0], [1280,0], [1280,960] ] )

    # img, out = perspective_transform(undistorted_img, src_points, dst_points )
    # cv2.imshow("perspective", out)
    #key = cv2.waitKey(1)
    
    
 
cv2.destroyAllWindows()
