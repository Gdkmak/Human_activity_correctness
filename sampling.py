import random
import numpy as np
import pandas as pd
from datetime import datetime


def downsampling(arr, lowest):
    """
    The function returns a sliced array, it downsampels all the array (movements) that have more than the lowest number of frame (28). 
    It downsamples the array and returns an array with shape (lowest_number_of_frames,number of joints coordinates) ex. (28,66).
    Parameters: 
    arr -- 2D array 
    lowest -- interger number represents the lowest number of frames among the movements 
    Returns: 
    new_arr -- 2D array with new shape (lowest, coordinates) 
    
    """
    fr = arr.shape[0]
    joints = arr.shape[1]
    step =int(np.floor(fr/lowest)) # calculate the steps needed for slicing
    arr_sliced = arr[::step,:]
    
    remainer = arr_sliced.shape[0]- lowest  # after the slcing there are some leftove
    
    # randomly select what left of the frames that need to delete to get the lowest number of frames
    rn_indices = []
    i = 0
    while(i != remainer):
        rn_idx = random.randint(0,lowest)
        if rn_idx not in rn_indices:
            rn_indices.append(rn_idx)
            i += 1
        else:
            continue

    new_arr = np.delete(arr_sliced, rn_indices, axis = 0) # delete them

    return new_arr


def upsampling(arr, countline, fr_max, inter_method):
    """
    The function converts the array to a time series dataframe, then it adds new NaN frames till the numbers of frames reachs the maximum.
    Then it replaces these NaN frames with values using the specified interpolation method. 
    Parameters: 
    arr -- 2D array
    countline -- integer number represents the number of frames in the array
    fr-max -- interger number represents the maximum number of frames among the movements
    inter_method -- the interpolation method used  
    Returns: 
    new_df -- a dataframe array  
    
    """
    
    # convert to time serise using pandas library 

    df = pd.DataFrame(data=arr[0:,0:],   # values
               index= [int(i*33) for i in range(countline)], columns=None) # 33 since the kinect configured to capture 30fps
    df.index = pd.to_datetime(df.index,unit= 'ms' )

    # add NaN frames and make the number of frames equal to the highest number in the Kinect case 179 frames
    i = 0
    while(i != fr_max-countline):
        rn_mil = random.randint(0,99)*10000 + random.randint(0,10)
        rn_s = random.randint(0,np.floor(countline/33))
        dt_t = datetime(1970, 1, 1, 0, 0, rn_s , rn_mil)
    
        if (dt_t in df.index): # make sure not to overlap exisitng ones 
            continue
        elif (df.index[countline-1] < dt_t): # make sure they are not outside the range
            continue
        else:
            df.loc[dt_t] = np.nan
            i +=1
            
    df = df.sort_index()
    # Select method according to the parameter (inter_method)
    if (inter_method == "back_fill"):
        new_df = df.bfill()
    elif (inter_method == "mean"):
        new_df = df.interpolate()
    else:
        new_df = df.interpolate(method=inter_method, order=3) 

    return new_df
