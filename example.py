"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import time
import numpy as np
from collections import deque
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

calib = True
start = time.time()
N = 10
xCenter = [] 
xRight = []
xLeft = []
yCenter = []
yTop = []
yBottom = []
xArr = np.ones(N)
yArr = np.ones(N)
pts = deque(maxlen=64)
i=0
while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    frame = cv2.flip(frame, 1)

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ''
    if type(gaze.horizontal_ratio()) is float:
        if(calib):
            now = time.time()-start
            if now < 3:
                text = 'Look  Top Right'
                xRight.append(gaze.horizontal_ratio())
                yTop.append(gaze.vertical_ratio())
                frame = cv2.circle(frame, (frame.shape[1], 0), 20, (0,0,255), -1)
            elif now >=3 and now < 6:
                text = 'Look Top Left'
                xLeft.append(gaze.horizontal_ratio())
                yTop.append(gaze.vertical_ratio())
                frame = cv2.circle(frame, (0, 0), 20, (0,0,255), -1)
            elif now >=6 and now < 9:
                text = 'Look Bottom Left'
                xLeft.append(gaze.horizontal_ratio())
                yBottom.append(gaze.vertical_ratio())
                frame = cv2.circle(frame, (0, frame.shape[0]), 20, (0,0,255), -1)
            elif now >=9 and now < 12:
                text = 'Look Bottom Right'
                xRight.append(gaze.horizontal_ratio())
                yTop.append(gaze.vertical_ratio())
                frame = cv2.circle(frame, (frame.shape[1], frame.shape[0]), 20, (0,0,255), -1)
            elif now >=12 and now < 15:
                text = 'Look Bottom Right'
                xCenter.append(gaze.horizontal_ratio())
                yCenter.append(gaze.vertical_ratio())
                frame = cv2.circle(frame, (round(frame.shape[1]/2), round(frame.shape[0]/2)), 20, (0,0,255), -1)
            else:
                calib = False

        else:
            xArr[i%N] = np.interp(gaze.horizontal_ratio(), [np.mean(xLeft), np.mean(xCenter), np.mean(xRight)], [0,0.5,1])
            yArr[i%N] = np.interp(gaze.vertical_ratio(), [np.mean(yTop), np.mean(yCenter), np.mean(yBottom)], [0,0.5,1])
            frame = cv2.circle(frame, (round(np.mean(xArr)*frame.shape[1]), round(np.mean(yArr)*frame.shape[0])), 30, (0,255,0), 2) 
            pts.appendleft((round(np.mean(xArr)*frame.shape[1]), round(np.mean(yArr)*frame.shape[0])))
            for j in range(1, len(pts)):
                thickness = int(np.sqrt(64 / float(j + 1)) * 2.5)
                cv2.line(frame, pts[j - 1], pts[j], (0, 0, 255), thickness)
        
        i+=1
    
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        
    cv2.imshow("window", frame)

    if cv2.waitKey(1) == 27:
        break
