#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:07:19 2023

@author: laurenparola
"""
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import quaternion
import numpy as np
from scipy.signal import butter, filtfilt,find_peaks
def rotate_imus(sensor_dict,data_freq):
    for sensor in sensor_dict:

        sensor_dict[sensor] = align_data(sensor_dict[sensor],sensor,data_freq)
    return sensor_dict

def find_standing(y_data,freq):
    
    #find jump peaks and walking start
    
    index = y_data.index

    peak_locs,_ = find_peaks(-y_data.values, height = -20)
   
    stand_range = [[peak_locs[i-1]+2*freq,peak_locs[i]-2*freq] for i in range(1,len(peak_locs)) if peak_locs[i] - peak_locs[i-1] > 3*freq]
    
    # if there is no break between hops and data, use before hops
    if len(stand_range)==0:
        stand_range = [0,peak_locs[0]-0.5*freq]
    else:
        stand_range = stand_range[0]

    return stand_range

def rotate_vectors_less_overhead(q, v):
    """Same as quaternion.rotate_vectors, without all the axis fanciness and checking"""
    m = np.array([
        [1.0 - 2*(q.y**2 + q.z**2), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w)],
        [2*(q.x*q.y + q.z*q.w), 1.0 - 2*(q.x**2 + q.z**2), 2*(q.y*q.z - q.x*q.w)],
        [2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1.0 - 2*(q.x**2 + q.y**2)]])
    return np.einsum('ij,...j', m, v)

def quaternion_conjugate(q):

	q_conj = np.zeros((q.shape))
	# print(q_conj) # FOR DEBUGGING

	q_conj[0] = q[0]
	q_conj[1] = -q[1]
	q_conj[2] = -q[2]
	q_conj[3] = -q[3]
	# print(q_conj) # FOR DEBUGGING

	return q_conj

def quaternion_multiply(q, r):

	n = np.zeros((q.shape))

	n[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
	n[1] = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
	n[2] = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
	n[3] = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]

	return n



def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix   

def vertical_orientation(data,key, freq,is_feet=False):
    #align to gravity where X is up 
    stand_range = find_standing(data[key]['Acc_X'],freq)
    vertical_or = data[key][['Acc_X','Acc_Y','Acc_Z']].iloc[int(stand_range[0]):int(stand_range[1])].mean(axis=0).values

    rot_mat = rotation_matrix_from_vectors(vertical_or, [9.8,0,0])


    return rot_mat,stand_range

def vertical_orientation_pelvis(data,key, freq,is_feet=False):

    stand_range = find_standing(data[key]['Acc_Y'],freq)
    vertical_or = data[key][['Acc_X','Acc_Y','Acc_Z']].iloc[int(stand_range[0]):int(stand_range[1])].mean(axis=0).values

    rot_mat = rotation_matrix_from_vectors(vertical_or, [0,9.8,0])


    return rot_mat,stand_range

def align_feet(data,key1,key2, freq):

    stand_range = find_standing(data[key1]['Acc_Y'],freq)
    foot1 = data[key1][['Acc_X','Acc_Y','Acc_Z']].iloc[int(stand_range[0]):int(stand_range[1])].mean(axis=0).values
    foot2 = data[key2][['Acc_X','Acc_Y','Acc_Z']].iloc[int(stand_range[0]):int(stand_range[1])].mean(axis=0).values
    rot_mat = rotation_matrix_from_vectors(foot1, foot2)


    return rot_mat,stand_range
def left_right_align(sensor_dict,key1, key2, freq):
        rotation_matrix, stand_range = align_feet(sensor_dict,key1,key2, freq)
        sensor_dict[key1]  = rotate_data(sensor_dict[key1],rotation_matrix)
        sensor_dict[key2]  = rotate_data(sensor_dict[key2],rotation_matrix)
        return sensor_dict
def calibration_sensor_data(sensor_dict,key, freq,is_feet=False):
	#calibrate sensor data to gravity
	
	#find a period of standing and calculate a rotation matrix between real data and ideal gravity orientation
        rotation_matrix, stand_range = vertical_orientation(sensor_dict,key, freq,is_feet)

        rotated_data  = rotate_data(sensor_dict[key],rotation_matrix)

        
        return rotated_data

def butter_low(data,fs, order, fc):
    '''
    Zero-lag butterworth filter for column data (i.e. padding occurs along axis 0).
    The defaults are set to be reasonable for standard optoelectronic data.
    '''
    # Filter design
    b, a = butter(order, 2*fc/fs, 'low')
    # Make sure the padding is neither overkill nor larger than sequence length permits
    padlen = min(int(0.5*data.shape[0]), 200)
    # Zero-phase filtering with symmetric padding at beginning and end
    filt_data = filtfilt(b, a, data, padlen=padlen, axis=0)
    return filt_data

def highpass_iir(data,fs, order, fc=.25 ):
    nyq = 0.5 * fs
    normal_cutoff = fc / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # Make sure the padding is neither overkill nor larger than sequence length permits
    padlen = min(int(0.5*data.shape[0]), 200)
    # Zero-phase filtering with symmetric padding at beginning and end
    filt_data = filtfilt(b, a, data, padlen=padlen, axis=0)
    return filt_data

def rotate_data(data,rotation_matrix):
        rot_quat= quaternion.from_rotation_matrix(rotation_matrix)
        temp_data = quaternion.rotate_vectors(rot_quat, data[['Acc_X','Acc_Y','Acc_Z']].values, axis=1)
        data[['Acc_X','Acc_Y','Acc_Z']]=temp_data

        temp_data = quaternion.rotate_vectors(rot_quat, data[['Gyr_X','Gyr_Y','Gyr_Z']].values, axis=1)
        data[['Gyr_X','Gyr_Y','Gyr_Z']]=temp_data

        temp_data = quaternion.rotate_vectors(rot_quat, data[['Mag_X','Mag_Y','Mag_Z']].values, axis=1)

        data[['Mag_X','Mag_Y','Mag_Z']]=temp_data
        quat_cols = ['Quat_q0','Quat_q1','Quat_q2','Quat_q3']
        rotated_ori = []
        for i in data[quat_cols].values:
            rotated_ori.append(quaternion_multiply(i, quaternion.as_float_array(rot_quat)))
        data[quat_cols] = rotated_ori
        
        return data

def rotate_data_from_quaternion(data, quat):
    quat_cols = ['Quat_q0','Quat_q1','Quat_q2','Quat_q3']
    rot_quat = quaternion.as_quat_array(quat)
    temp_data = quaternion.rotate_vectors(rot_quat, data[['Acc_X','Acc_Y','Acc_Z']].values, axis=1)
    data[['Acc_X','Acc_Y','Acc_Z']]=temp_data

    temp_data = quaternion.rotate_vectors(rot_quat, data[['Gyr_X','Gyr_Y','Gyr_Z']].values, axis=1)
    data[['Gyr_X','Gyr_Y','Gyr_Z']]=temp_data

    temp_data = quaternion.rotate_vectors(rot_quat, data[['Mag_X','Mag_Y','Mag_Z']].values, axis=1)
    data[['Mag_X','Mag_Y','Mag_Z']]=temp_data
    rotated_ori = []
    for i in data[quat_cols].values:
        rotated_ori.append(quaternion_multiply(i, quat))

    data[quat_cols] = rotated_ori
    return data
import math
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z     
    
def find_rotations_from_pelvis(sensor_dict, sensor, stand_range):
    quat_cols = ['Quat_q0','Quat_q1','Quat_q2','Quat_q3']
    rotation = quaternion_multiply(quaternion_conjugate(sensor_dict[sensor][quat_cols].iloc[int(stand_range[0]):int(stand_range[1])].mean(axis=0).values), sensor_dict['Pelvis'][quat_cols].iloc[int(stand_range[0]):int(stand_range[1])].mean(axis=0).values)
    return rotation
def sensor_to_body_cal(sensor_dict, data_freq):
    rotation_matrix, stand_range = vertical_orientation_pelvis(sensor_dict,'Pelvis', data_freq,)
    rotation_dict = {}

    #find rotation of each sensor to pelvis orientation
    for sensor in sensor_dict:
        print(sensor)
        z = find_rotations_from_pelvis(sensor_dict, sensor, stand_range)
        print(euler_from_quaternion(z[0],z[1],z[2],z[3]))
        rotation_dict[sensor] = find_rotations_from_pelvis(sensor_dict, sensor, stand_range)

    for sensor in rotation_dict:
        sensor_dict[sensor] = rotate_data_from_quaternion(sensor_dict[sensor], rotation_dict[sensor])
    return sensor_dict
        

    #rotate each
        

def align_data(df,position,data_freq):
    '''
    Aligns IMU to standard coordinate system
    X - forward; Y - up; Z - right
    rotation differs depending on  sensor placement
    '''

    df_old=df.copy()
    #rotation that has to be applied depends on the sensor position and side
    if 'foot' in position and 'Pelvis' not in position:

        x_rot_quat = quaternion.from_euler_angles(-np.pi/2,0,0)
        #z_rot_quat = quaternion.from_euler_angles(0, 0, -np.pi)
        rot_quat = x_rot_quat #*z_rot_quat
    #if 'Left foot' in position:
        #x_rot_quat = quaternion.from_euler_angles(np.pi/2,0,0)
        #z_rot_quat = quaternion.from_euler_angles(0, 0, np.pi)
        #rot_quat = x_rot_quat*z_rot_quat
        
    if 'Right' in position and 'foot' not in position:
        z_rot_quat = quaternion.from_euler_angles(0, 0, -np.pi/2)
        rot_quat = z_rot_quat

        
    if 'Left' in position and 'foot' not in position:
        z_rot_quat = quaternion.from_euler_angles(0, 0, np.pi/2)
        y_rot_quat = quaternion.from_euler_angles(0, np.pi, 0)
        rot_quat = z_rot_quat*y_rot_quat

    if position == 'Pelvis':

        y_rot_quat = quaternion.from_euler_angles( 0,np.pi/2,0)
        x_rot_quat = quaternion.from_euler_angles(np.pi/2,0,0)
        rot_quat = y_rot_quat*x_rot_quat
    #rotate each accelerometer and gyro data then save back into the original dataframe
    accel_col = [col for col in df.columns if 'Acc' in col]
    gyro_col = [col for col in df.columns if 'Gyr' in col]
    mag_col = [col for col in df.columns if 'Mag' in col]
        
    #ori_col = [col for col in df.columns if 'Ori' in col]


    df[gyro_col] = quaternion.rotate_vectors(rot_quat, df[gyro_col], axis=1)

    df[accel_col] = quaternion.rotate_vectors(rot_quat, df[accel_col], axis=1)

    df[mag_col] = quaternion.rotate_vectors(rot_quat, df[mag_col], axis=1)

    quat_cols = ['Quat_q0','Quat_q1','Quat_q2','Quat_q3']
    rotated_ori = []
    for i in df[quat_cols].values:
        rotated_ori.append(quaternion_multiply(i, quaternion.as_float_array(rot_quat)))
    df[quat_cols] = rotated_ori
        

    return df

def upload_sensor_txt(file_path,activity):
    #upload sensors data as pandas dataframes using sensor names
    df_list = []
    with open(os.path.join(file_path,'sensor_name.txt')) as f:
        data = f.read()
    split = [x for x in data.split()]

    
    temp_list = []
    name_list = []
    code_list = []
    for i in split:
        if 'B4' in i:
            name = ' '.join(temp_list)
            if 'Helvetica;}' in name:
                name= ' '.join(name.split(' ')[1:])
            name_list.append(name)
            code_list.append(i)
            temp_list = []
        if 'B4' not in i and '\\' not in i:
            temp_list.append(i)
    
    #labels = pd.read_csv(os.path.join(file_path,'sensor_name.txt'), sep='\s+',names=['Label','ID'],dtype=str
    
    walking_path = os.path.join(file_path,activity) 
    sensor_list = {}
    for code,name in zip(code_list,name_list):
        
        sensor_name = [sensor for sensor in os.listdir(walking_path) if code in sensor][0]
        sensor_list[name] = pd.read_table(os.path.join(walking_path,sensor_name), sep="\t",  skiprows=4)

    return sensor_list

