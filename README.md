# Project aim:

The aim, here, is to investigate how to yield optimal results in this field by trying various different approaches at every stage of skeleton data processing; as a consequence of this work, HAR might take a significant step forward.
 
In order to achieve this aim, we will be using an open access dataset called UI-PRMD. This data was collected by Idaho University using a Vicon optical tracker, and a Kinect camera. The data provides the angles and positions of body joints in relation to patients (subjects) who were undergoing physical therapy and rehabilitation programs which required the undertaking of common exercises such as deep squats, standing shoulder abduction etc.









	Drawing New Joint coordinates on the Trajectory Based on a Threshold:

This possible approach is to draw new frame(s) between two existing frames by using 3D interpolation. The question here how to draw these frame(s), which consists of joints coordinates x,y and z? The answer here is using the trajectory between two consecutive frames. When an object with mas moves from point A to point B, it follows a path through space. This path is called the trajectory. 

This suggests to locate and draw new joints coordinates on the trajectory if only the difference between two consecutive frames exceeds a certain threshold. The difference will be called gap. The gap is the absolute difference between the summation of all x coordinates in two consecutive frames. i is the number of frame in the gesture and j is the number of coordinates in the same gesture. Please refer to the notation below. 

〖gap〗_i=| ∑_(j=1)^n▒X_ij - ∑_(j=1)^n▒〖X_((i+1)j)  〗|

The average of gaps in each gesture will be taken. The same process will be repeated for all gestures from the same classification in the same movement. This process will result in a scalar. For example, if I have 100 incorrect gestures and 100 correct gestures in movement number 5. After calculating the average of gaps in each gesture, we will end up with a scalar of 100 elements for incorrect gestures and another scalar with 100 elements for correct gesture. The average of each scalar is the threshold of each classification in the movement number 5.  

Now we calculated the gap and the threshold, what are they useful for? Now we go again to the gesture and check if a gap exceeds a threshold, then joints coordinates values will be polynomially interpolated to draw a new frame(s) on a trajectory between two frames. 

 
Figure 5 – Interpolated Trajectory - Elbow flexion


The figure 5 shows two consecutive frames in elbow flexion gesture. The gap between the frames in the right arm space is relatively big and exceeded the threshold. Therefore, we filled this gap with a body coordinates and hence a frame. 
