import numpy as np
import glob

def subtraction(frame_sum):
    """
    The function returns the result of substraction the summation of two consecutive frames. 
    """
    subtract_frames = []
    for i in range(len(frame_sum) - 1): 
        substract = abs(frame_sum[i] - frame_sum[i + 1])
        subtract_frames.append(substract)
    return subtract_frames


def thresholds(path):
    """
    This functions calculates the average of differences for each two consecutive frames in each sequence.
    Returns: 
    A list of thesholds
    """

    files = [f for f in glob.glob(path, recursive=True)]  # read all files in the path

    full_move= []
    avg_lst = []
    for f in (files):
        coordinates = []  # list of coordinates, then it will be transformed to 3D matrix
        file = open(f,"r")
        filecontent = file.readlines()
        for line in filecontent:
            current_line = line.split(",")
            for coordinate in range (len(current_line)):
                if (current_line[coordinate]) == "":
                    break
                else:
                    coordinates.append(float(current_line[coordinate]))
        coordinates = np.array(coordinates)
        arr = coordinates.reshape(len(filecontent),len(current_line))
        # calculate the summation of each frame
        frame_sum = np.sum(arr[:,::3],axis=1).tolist()
        # calculates the differences of every consecutive two frames
        difference = subtraction(frame_sum)
        # 3 avg of the substraction
        avg_lst.append(np.average(difference))
    
    # create a list of threshold for each movement (coorect and incorrect)    
    all_mov_avg = [] 
    step = 100
    for i in range(int(len(avg_lst)/step)):
        start = i*step
        stop = start + step
        all_mov_avg.append(np.average(avg_lst[start:stop]))
    return all_mov_avg
