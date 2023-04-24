import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import scipy

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

def find_mid_stance(gyro,data_freq):
    mid_stance_time,_ = find_peaks(gyro,height=[-2,0],distance=data_freq*.8)
    toe_off, _ = find_peaks(-gyro,height=3,distance=data_freq*.8)
    heel_contact, _ = find_peaks(-gyro,height=[0,3],distance=data_freq*.8)

    mid_stance_time = [mid_stance_time[mid_stance_time > i][0] for i in toe_off ]

    plt.plot(gyro)
    print('yes')
    plt.scatter(mid_stance_time,gyro[mid_stance_time],marker='o')
    plt.show()

    return mid_stance_time
    
import matplotlib.pyplot as plt        
def find_velocity(gyro, tangent_acc,normal_acc,data_freq):
    mid_stance_time = find_mid_stance(gyro,data_freq)

    velocity_list = []

    for i in range(1,len(mid_stance_time)):
        start = mid_stance_time[i-1]
        end = mid_stance_time[i]

        ang = scipy.integrate.cumulative_trapezoid(gyro[start:end],dx=1/data_freq,axis=0, initial=0)

        y = np.expand_dims(np.matrix([tangent_acc[start:end],normal_acc[start:end]]), axis=1)

        A = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
        B = y

        res_mat = np.subtract(np.einsum('mnr,ndr->mdr', A, B),np.dstack([[0,-9.8]]*(end-start)).transpose(1,0,2))
             
        forward_acc = res_mat[0,:,:]
        vertical_acc = res_mat[1,:,:]

        

        forward_vel = scipy.integrate.cumulative_trapezoid(forward_acc,dx=1/data_freq, axis=1, initial=0)

        vertical_vel = scipy.integrate.cumulative_trapezoid(vertical_acc,dx=1/data_freq, axis=1, initial=0)

        forward_dis = scipy.integrate.trapezoid(forward_vel,dx=1/data_freq)- 0.5*(len(ang)/data_freq)*forward_vel[0,len(ang)-1]
        
        vertical_dis = scipy.integrate.trapezoid(vertical_vel,dx=1/data_freq) - 0.5*(len(ang)/data_freq)*vertical_vel[0,len(ang)-1]
        step_length = np.sqrt(forward_dis**2 + vertical_dis**2)

        velocity = step_length /(len(ang)/data_freq)
        velocity_list.append(velocity)

    return velocity_list




