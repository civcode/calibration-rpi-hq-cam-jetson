import cv2
import numpy as np
import os
import glob
import pickle
import sys

# sensor_id=1 ... left camera
# sensor_id=0 ... right camera
sensor_id = 0
assert(sensor_id == 0 or sensor_id == 1)


# File for captured image
if sensor_id == 1:
    file_path = './img_left/'
    data_name = '/calib_left.dat'
else:
    file_path = './img_right/'
    data_name = '/calib_right.dat'

if not os.path.exists(file_path):
    print("\nPath " + str(file_path) + " does not exist.")
    print("No images found.")
    quit()

# Displayed image size
scale_factor = 0.5

# Calibration pattern parameters
CHECKERBOARD = (6,9)
#CHECKERBOARD = (7,10)
square_size = 19.8e-2 / 8

#corner_subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
corner_subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)


objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float32)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(file_path + '/*.png')
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
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners, (3,3), (-1,-1), corner_subpix_criteria)
        imgpoints.append(corners)
        
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        #cv2.imshow('img', img)
        img_scaled = cv2.resize(img, (img_heigth, img_width), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('img', img_scaled)
        cv2.waitKey(500)
        
        
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

print('Image size:', gray.shape[::-1])
#print(objpoints)

img_size = (w, h)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# store data
calib_data = {"dim": _img_shape[::-1], "K": mtx, "D": dist}
pickle_file = os.getcwd() + data_name
with open(pickle_file, "wb") as f:
    pickle.dump(calib_data, f)


#new_K, valid_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h, w), 1, (h,w))
#new_K, valid_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h, w), 1, centerPrincipalPoint=True)
new_K, valid_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h, w), 1, centerPrincipalPoint=False)

print("\nrms=" + str(ret))
print("\nCamera matirx=")
print(mtx)
print("\nDistortion parameters=")
print(dist)
print("\nValid roi:")
print(valid_roi)

for fname in images:
    img = cv2.imread(fname)
    w,h = img.shape[:2]
    #undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_K)
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

