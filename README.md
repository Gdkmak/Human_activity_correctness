# Evaluating Pre-processing Methods with Deep Learning Algorithms on Human Activity Skelton Data Obtained from Kinect and Vicon Sensors

##### The aim, here, is to investigate how to yield optimal results in this field by trying various different approaches at every stage of skeleton data processing; as a consequence of this work, HAR might take a significant step forward. This reposotry tries to tackle the issue of variable sized inputs. 

## Dataset Used

##### In order to achieve this aim, I will be using an open access dataset called [UI-PRMD](https://webpages.uidaho.edu/ui-prmd/). This data was collected by Idaho University using a Vicon optical tracker, and a Kinect camera. The data provides the angles and positions of body joints in relation to patients (subjects) who were undergoing physical therapy and rehabilitation programs which required the undertaking of common exercises such as deep squats, standing shoulder abduction etc.


#### Invented approach is based on drawing New Joint coordinates on the Trajectory Based on a Threshold:

This possible approach is to draw new frame(s) between two existing frames by using 3D interpolation. The question here how to draw these frame(s), which consists of joints coordinates x,y and z? The answer here is using the trajectory between two consecutive frames. When an object with mas moves from point A to point B, it follows a path through space. This path is called the trajectory. 

This suggests to locate and draw new joints coordinates on the trajectory if only the difference between two consecutive frames exceeds a certain threshold. The difference will be called gap. The gap is the absolute difference between the summation of all x coordinates in two consecutive frames. i is the number of frame in the gesture and j is the number of coordinates in the same gesture. Please refer to the notation below. 

`〖gap〗_i=| ∑_(j=1)^n▒X_ij - ∑_(j=1)^n▒〖X_((i+1)j)  〗|`
