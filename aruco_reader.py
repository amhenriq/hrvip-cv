import cv2
import cv2.aruco as aruco
import datetime
import numpy as np
import math
 

########################################################################################
cap = cv2.VideoCapture(1) # 1 if external camera, 0 if webcam
fps = cap.get(5) # print(fps) will tell you how many fps. The logitech cam is 25 frames per second
start_time = datetime.datetime.utcnow().timestamp() 
file = open("datafile.csv", "w") #saves time, x, y, z position
file2 =  open("rvec.csv", "w") #saves rotation values for roll, pitch, yaw
marker_size = 0.0655 #meters. CHANGE THIS VALUE IF USING NEW ARUCO MARKER. This is the length of one side (I measured with a caliper for accuracy)
video_width = 1280 #pixels. Adjust this value depending on the camera
video_height = 960 #pixels. Adjust this value depending on the camera
########################################################################################

########################################################################################
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mov',fourcc, fps, (video_width , video_height), True) #you can rename video file to something other than output.mov
########################################################################################

# load in camera matrix parameters for pose
with np.load('calib.npz') as X: #calib.npz is the file saved from calibration.py
    mtx, dist,rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame.shape) #gives video pixel dimenions. In this case, it is 960x1280 rgb 
    
    start_t = cv2.getTickCount(); #start_t could be useful for video delays
    
    
    frame = cv2.GaussianBlur(frame, (11,11), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) #6X6 250 very common 
    parameters =  aruco.DetectorParameters_create()
    #print(parameters)
 
    try:
        #lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        #print(corners)
     
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
        #The second parameter is the size of the marker side in a unit. Translation vectors of the estimated poses will be in the same unit
        # rvec is the rotation matrix
        #tvec is the translation matrix 
        rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[0], marker_size, mtx, dist) 
        #print(tvec[0][0])
        
        aruco.drawAxis(frame, mtx, dist, rvec[0], tvec[0], marker_size) #Draw Axis

        #write w y z translation to datafile.csv
        now = datetime.datetime.utcnow().timestamp() - start_time
        output = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}\n".format(now, tvec[0][0][0], tvec[0][0][1], tvec[0][0][2])
        file.write(output)
        
        #write rotation values for roll, pitch, yaw into rvec.csv
        output2 = "{0} {1} {2}\n".format(rvec[0][0][0], rvec[0][0][1], rvec[0][0][2])
        file2.write(output2)
         
    #write time, translation, and rotation as not-a-number if the camera cannot detect aruco marker 
    except Exception as E:
        #print(E)
        now = datetime.datetime.utcnow().timestamp() - start_time
        output = "{0} {1} {2} {3}\n".format(now, np.nan, np.nan, np.nan)
        file.write(output)
        
        #write all three dimensions
        output2 = "{0} {1} {2}\n".format(np.nan, np.nan, np.nan)
        file2.write(output2)
############### COME BACK TO THE VIDEO WRITING    
    # Display result
    cv2.imshow('Detected Aruco Markers',frame)
    
    
    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;
    
    
    key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF;
    
    if cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF == ord('0'):
        out.write(frame) 
#    if cv2.waitKey(40) & 0xFF == ord('0'):
        break
# When everything done, release the capture
    
file.close()
file2.close()
cap.release()
out.release()
cv2.destroyAllWindows()