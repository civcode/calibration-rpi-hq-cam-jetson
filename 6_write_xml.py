import cv2
import glob
import numpy as np
import os
import pickle
import sys


file_names =['calib_left.dat', 'calib_right.dat']

def writeCalibrationData(file_name):
    
    data_path = os.getcwd() + "/" + file_name
    print("Reading file")
    print(data_path)

    if not os.path.exists(data_path):
        print("Path des not exist.")
        print(data_path)
        sys.exit()

    with open(data_path, "rb") as f:
        calib_data = pickle.load(f)

    print('dim =', calib_data["dim"])
    print('K =', calib_data["K"])
    print('D =', calib_data["D"])

    if file_name.find('left') >= 0:
        xml_file_name = 'calib_left_data.xml'
    else:
        xml_file_name = 'calib_right_data.xml'

    print("Writing xml file")
    print(xml_file_name)
    s = cv2.FileStorage(xml_file_name, cv2.FileStorage_WRITE)
    s.write('cameraMatrix', calib_data["K"]) 
    s.write('distCoeffs', calib_data["D"]) 
    s.release()
    

writeCalibrationData(file_names[0])
writeCalibrationData(file_names[1])


