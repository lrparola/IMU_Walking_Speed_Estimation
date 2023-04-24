import os
import pandas as pd
import matplotlib.pyplot as plt
import IMU_event_clean as tmp
import Upload_XSENS_textfiles as pp
import shank_speed_utils as spd
import numpy as np
import pendulum_utils as pu
import quaternion
import shank_walking_utils as sw

file_path = '/Users/laurenparola/Library/CloudStorage/Box-Box/Lauren-Projects/Code/-_-/Test_Data_2_17/IMU/'

#input the activity folder, mine is named walking 
sensor_data = pp.upload_sensor_txt(file_path,'Treadmill Walking')

data_freq = 100
shank_length = 0.44
thigh_length = 0.51
leg_length = shank_length+thigh_length

#calibrate all sensors to pelvis IMU
sensor_data = pp.sensor_to_body_cal(sensor_data, data_freq)


for sensor in sensor_data:
        #calibrate imu data to gravity (x is up)
       sensor_data[sensor] = pp.calibration_sensor_data(sensor_data,sensor, data_freq)
shank_data = ['Left shank','Right shank']

#filter the IMU data
for sensor in shank_data:
    sensor_data[sensor][['Gyr_X','Gyr_Y','Gyr_Z']] = pp.butter_low(sensor_data[sensor][['Gyr_X','Gyr_Y','Gyr_Z']],data_freq, 4, 14)
    sensor_data[sensor][['Acc_X','Acc_Y','Acc_Z']] = pp.butter_low(sensor_data[sensor][['Acc_X','Acc_Y','Acc_Z']],data_freq, 4, 2)

#sensor_data = pp.sensor_to_body_cal(sensor_data, data_freq)
velocity_result = sw.find_velocity(sensor_data['Left shank']['Gyr_Y'].values, sensor_data['Left shank']['Acc_Z'].values,sensor_data['Left shank']['Acc_X'].values,data_freq) 

#print average walking speed velocity
print(str(np.mean(velocity_result))+'+/-'+str(np.std(velocity_result)))

