
sensor_id = 0
assert(sensor_id == 0 or sensor_id == 1)

# File for captured image
if sensor_id == 0:
    file_path = './img_left/'
    data_name = '/calib_left.dat'
    data_fisheye_name = '/calib_fisheye_left.dat'
else:
    file_path = './img_right/'
    data_fisheye_name = '/calib_fisheye_right.dat'

file_prefix = 'img_'
file_suffix = '.png'

# Displayed image size
scale_factor = 2.0