from ast import increment_lineno
import cv2
import glob
import numpy as np
import os
import pickle

cam_num = 0
assert(cam_num == 0 or cam_num == 1)

# Displayed image size
scale_factor = 0.5

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

balance = 0.2 #0.5
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)

images = glob.glob(file_path + '/*.png')


print("\nPress p to increase balance value")
print("Press m to decrease balance value")

for fname in images:
    img = cv2.imread(fname)
    #print(img.shape)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    w,h = img.shape[:2]
    img_width = int(w*scale_factor)
    img_heigth = int(h*scale_factor)
    
    
    img_scaled = cv2.resize(img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
    undistorted_scaled = cv2.resize(undistorted_img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
    
    #cv2.imshow('img', img)
    #cv2.imshow("undistorted", undistorted_img)
    cv2.imshow('img', img_scaled)
    cv2.imshow("undistorted", undistorted_scaled)
    
    #print('image')
    key = cv2.waitKey(0)
    if key == ord("q"):
        quit()
    while key == ord("p") or key == ord("m"):
        increment = 0.1
        if key == ord("p"):
            if balance <= 1.0-increment:
                balance += increment 
                print('balance: {0:.2f}'.format(balance))
        elif key == ord("m"):
            if balance >= 0.0+increment:
                balance -= increment
                print('balance: {0:.2f}'.format(balance))
                
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
        #undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        undistorted_scaled = cv2.resize(undistorted_img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
        #cv2.imshow("undistorted", undistorted_img)
        cv2.imshow("undistorted", undistorted_scaled)
        key = cv2.waitKey(0)
        
    
    
    
    
 
cv2.destroyAllWindows()
