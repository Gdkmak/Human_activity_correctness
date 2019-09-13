import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
import glob
from tqdm import tqdm
import random
import pandas as pd

import sampling as sam
import threshold_utils as tresh


import importlib
importlib.reload(sam)
importlib.reload(tresh)


def frame_max_min (path, plot= "None"):
    """
    Parameters:
    path -- location of the dataset
    Return:
    the highest number of frames in the movement 
    """
    
    mov_frames_size = []
    files = [f for f in glob.glob(path, recursive=True)]
    for f in files:
        file = open(f,"r")
        mov_frames_size.append(len(file.readlines()))
    if (plot == "hist"):
        plt.hist(mov_frames_size,bins='auto')
        plt.title("Histogram of # of frames")
        plt.show()
    return np.max(mov_frames_size), np.min(mov_frames_size)

def read_files (path, inter_method ="polynomial" , sampling_method = "downsampling"):
    """
    This function reads and intepolates data according to the method mentioned in the parameter.
    Parameters:
    path -- location of the dataset
    fr_max -- output of the frame_max() 
    inter_method -- the method of the interpolation Linear, polynomial, pad, mean, back filling etc..
    Returns: 
    X -- 3D numpy array (samples, Height, Width)
    Y -- 2D numpy array (samples, #Subject #Movement #Status) Status: correct 1, incorrect 0   
    """
    
    #joint_coo = 66 # number of joints coordinate, in kinect is 22*3 = 66
    files = [f for f in glob.glob(path, recursive=True)]  # read all files in the path
    countfile = 0
    countline = 0

    fr_max, fr_min = frame_max_min(path)
    
    # creat Y label matrix 
    Y = [[0 for c in range (3)] for r in range(len(files))] 
    full_move= []
    for f in tqdm(files):
        coordinates = []  # list of coordinates, then it will be transformed to 3D matrix
        file = open(f,"r")
        Y[countfile][0] = int(f.rsplit('\\',1)[1].split('.')[0].split('_')[1].split('_')[0][1::]) # subject number 
        Y[countfile][1] = int(f.rsplit('\\',1)[1].split('.')[0].split('_')[0][1::]) # movement number
        if "Positions_inc" not in f.rsplit('\\',1)[0]:
            Y[countfile][2] =  1 # correct 
        else:
            Y[countfile][2] =  0 # incorrect
        countfile +=1
        
        filecontent = file.readlines()
        for line in filecontent:
            current_line = line.split(",")
            countline +=1 
            for coordinate in range (len(current_line)):
                if (current_line[coordinate]) == "":  # there are some extra lines at the end of the file, to make sure they are excluded
                    break
                else:
                    coordinates.append(float(current_line[coordinate]))
               

        coordinates = np.array(coordinates)
       
        arr = coordinates.reshape(len(filecontent), len(current_line))
   
        if (sampling_method == "downsampling"):
            full_move.append(sam.downsampling(arr, fr_min))           
        elif (sampling_method == "upsampling"):
            full_move.append(sam.upsampling(arr, arr.shape[0], fr_max, inter_method).values)
    X = np.asarray(full_move)
    Y = np.array(Y)

    return X, Y

def read_files_threshold(path):
    """
    This function calculates a threshold between frame for each movement; then interpolates frames when the threshold is exceeded.
    Parameters:
    path -- location of the dataset
    Returns: 
    X -- 3D numpy array (samples, Height, Width)
    Y -- 2D numpy array (samples, #Subject #Movement #Status) Status: correct 1, incorrect 0   
    """

    files = [f for f in glob.glob(path, recursive=True)]  # read all files in the path
    countfile = 0
    fr_max, fr_min = frame_max_min(path)
    
    threshold = tresh.thresholds(path)  # get the threshold between frames for each movement
    print("threshold: ", threshold) 
   
    # creat Y label matrix 
    Y = [[0 for c in range (3)] for r in range(len(files))] 
    full_move= []
    for f in tqdm(files):
        coordinates = []  # list of coordinates, then it will be transformed to 3D matrix
        file = open(f,"r")
        Y[countfile][0] = int(f.rsplit('\\',1)[1].split('.')[0].split('_')[1].split('_')[0][1::]) # subject number 
        Y[countfile][1] = int(f.rsplit('\\',1)[1].split('.')[0].split('_')[0][1::]) # movement number
        if "Positions_inc" not in f.rsplit('\\',1)[0]:
            Y[countfile][2] =  1 # correct 
        else:
            Y[countfile][2] =  0 # incorrect

        filecontent = file.readlines()
        for line in filecontent:
            current_line = line.split(",")
            for coordinate in range (len(current_line)):
                if (current_line[coordinate]) == "":  # there are some extra lines at the end of the file, to make sure they are excluded
                    break
                else:
                    coordinates.append(float(current_line[coordinate])) 

        coordinates = np.array(coordinates)
        arr = coordinates.reshape(len(filecontent), len(current_line))
           
        # get the threshold of the movement being processed 
        if (Y[countfile][2] == 1):
            mov_threshold = threshold[Y[countfile][1]-1]
        elif (Y[countfile][2] == 0):        
            mov_threshold = threshold[(Y[countfile][1]-1) + int((len(files)/200))] # 200 comes from the number of episodes in each movement     

        # calculate the summation of X coordinates of each frame
        frame_sum = np.sum(arr[:,::3],axis=1).tolist()
        # calculates the differences of every consecutive two frames
        difference = tresh.subtraction(frame_sum)
        
        ii = 0 
        for i in range(len(difference)):
            if ((len(filecontent)+ii+2) > fr_max):
                break
            if (difference[i] >  mov_threshold):
                for j in range (int(difference[i] / mov_threshold)):
                    if (ii==0):
                        arr = np.insert(arr, i+1, np.nan, axis=0)
                        ii+=1
                    else:
                        if((len(filecontent)+ii+2) > fr_max):
                            break
                        else:
                            arr = np.insert(arr, i+ii+1 , np.nan, axis=0)
                            ii+=1
        leftovers = fr_max - arr.shape[0] # calculate how many frames missing to maximum

        ran_ls = []
        for j in range(leftovers):
            ran_ls.append(random.randint(1,arr.shape[0]-1)) 

        arr = np.insert(arr, ran_ls, np.nan, axis=0) # add the leftovers 

        df = pd.DataFrame(data=arr[0:,0:])
        df.index = pd.to_datetime(df.index)

        new_df = df.interpolate(method="polynomial", order=3)

        full_move.append(new_df.values)
        countfile +=1

    X = np.asarray(full_move)
    Y = np.array(Y)
    return X, Y

def read_files_lowest (path):
    
    """ 
    This function returns a fixed number of frames array (samples,28,66), it will be used as a baseline or reference.
    Parameters:
    path -- location of the dataset
    num_frame -- in kinect case is 28 frames only 
    num_joints -- in kinect is 22x3
    Returns: 
    X -- 3D numpy array (samples, Height, Width)
    Y -- 2D numpy array (samples, #Subject #Movement #Status) Status: correct 1, incorrect 0   
    """
    
    _, fr_min = frame_max_min(path)
 
    files = [f for f in glob.glob(path, recursive=True)]  # read all files in the path
    countfile = 0
    countword = 0
     
    Y = [[0 for c in range (3)] for r in range(len(files))] 
    full_move= []
    for f in tqdm(files):
        countline = 0
        coordinates = []  # list of coordinates, then it will be transformed to 3D matrix later
        file = open(f,"r")
        Y[countfile][0] = int(f.rsplit('\\',1)[1].split('.')[0].split('_')[1].split('_')[0][1::]) # subject number 
        Y[countfile][1] = int(f.rsplit('\\',1)[1].split('.')[0].split('_')[0][1::]) # movement number
        if "Positions_inc" not in f.rsplit('\\',1)[0]:
            Y[countfile][2] =  1
        else:
            Y[countfile][2] =  0
        countfile +=1
        
        for line_num in range(fr_min):
            lines = file.readline()
            current_line = lines.split(",")
            countword = 0 
            for coordinate in range (len(current_line)):
                if (current_line[coordinate]) == "":
                    break
                else:
                    coordinates.append(float(current_line[coordinate]))
                    countword += 1
        full_move.append(coordinates)
    
    X = np.array(full_move).reshape(len(files),fr_min,countword)
    Y = np.array(Y)
    
    return X, Y


def train_test(X, Y):
    """
    This function splits data to training and test set. It uses cross subject manner, where every movement has different split for test
    and train. i.e. in movement 1 the 9th subject is used for test, while in movement 2 the 5th subject is used for test and so on.    
    Parameters:
    X -- 4D array   
    Y -- label array
    num_move -- number of movements in this case we have got 10
    Returns: 
    X_train_o -- training set for sampels  
    Y_train_o -- training set for labels 
    X_test_o -- training set for samples
    Y_test_o -- test set for labels
    
    """
   
    all_subject = np.arange(start= 1,stop= 11, dtype=int)
    num_mov = len(set(Y[:,1]))

    for m in range (num_mov):
                      
        ran_test = np.random.choice(all_subject, 1, replace=False).tolist() # number one indicate to get only one subject
        test_idx =  np.where((Y[:,0].astype(int) == ran_test) & (Y[:,1].astype(int) == m+1))
        train_idx = np.where((Y[:,0].astype(int) != ran_test) & (Y[:,1].astype(int) == m+1))
        if m+1 == 1 :
            X_train_o = X[train_idx]
            Y_train_o =  Y[train_idx] 
            X_test_o = X[test_idx]
            Y_test_o = Y[test_idx]
        else:
            X_train_o = np.append(X_train_o, X[train_idx], axis= 0) 
            Y_train_o =  np.append(Y_train_o, Y[train_idx], axis= 0) 
            X_test_o = np.append(X_test_o, X[test_idx], axis= 0)
            Y_test_o = np.append(Y_test_o, Y[test_idx], axis= 0)
    return X_train_o, Y_train_o,X_test_o, Y_test_o


def convert_to_one_hot(Y, C):
    """
    returns: 
    A 2d matrix with shape (num of classes,num of classes) 2 in this case.  
   
    """
    
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y




