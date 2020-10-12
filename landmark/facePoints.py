

import cv2
import numpy as np

# This below mehtod will draw all those points which are from 0 to 67 on face one by one.
def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=False, npy=False):
    points = []
    for i in range(startpoint, endpoint+1):
        if not npy:
            point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
        else:
            point = [faceLandmarks[i][0], faceLandmarks[i][1]]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

# Use this function for 70-points facial landmark detector model
# we are checking if points are exactly equal to 68, then we draw all those points on face one by one
def facePoints(image, faceLandmarks, npy=False):
    if not npy:
        assert(faceLandmarks.num_parts == 68)
    else:
        assert(faceLandmarks.shape[0] == 68)
    drawPoints(image, faceLandmarks, 0, 16, npy=npy)                     # Jaw line
    drawPoints(image, faceLandmarks, 17, 21, npy=npy)                    # Left eyebrow
    drawPoints(image, faceLandmarks, 22, 26, npy=npy)                    # Right eyebrow
    drawPoints(image, faceLandmarks, 27, 30, npy=npy)                    # Nose bridge
    drawPoints(image, faceLandmarks, 30, 35, isClosed=True, npy=npy)     # Lower nose
    drawPoints(image, faceLandmarks, 36, 41, isClosed=True, npy=npy)     # Left eye
    drawPoints(image, faceLandmarks, 42, 47, isClosed=True, npy=npy)     # Right Eye
    drawPoints(image, faceLandmarks, 48, 59, isClosed=True, npy=npy)     # Outer lip
    drawPoints(image, faceLandmarks, 60, 67, isClosed=True, npy=npy)     # Inner lip

# Use this function for any model other than
# 70 points facial_landmark detector model
def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=4):
    for p in faceLandmarks.parts():
        cv2.circle(im, (p.x, p.y), radius, color, -1)