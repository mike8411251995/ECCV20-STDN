import os
import dlib, cv2
import numpy as np
from landmark.facePoints import facePoints
from tqdm import tqdm

def writeFaceLandmarksToNPY(faceLandmarks, fileName):
    faceLandmarksArray = np.zeros((68, 2))
    for i in range(len(faceLandmarks.parts())):
        p = faceLandmarks.parts()[i]
        faceLandmarksArray[i][0] = int(p.x)
        faceLandmarksArray[i][1] = int(p.y)

    if not os.path.exists(os.path.dirname(fileName)):
        os.makedirs(os.path.dirname(fileName))
    
    np.save(fileName, faceLandmarksArray)
    
    return faceLandmarksArray

def detectLandmark(img_path, model_path='landmark/shape_predictor_68_face_landmarks.dat',
                   saveNPY=True, saveImg=False):
    frontalFaceDetector = dlib.get_frontal_face_detector()
    faceLandmarkDetector = dlib.shape_predictor(model_path)

    img = cv2.imread(img_path)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = frontalFaceDetector(imgRGB, 0)
    if len(face) == 0:
        tqdm.write(f'{img_path}: no face detected')
        return None

    face = face[0]
    landmarks = []

    faceRectangleDlib = dlib.rectangle(int(face.left()),int(face.top()),
                                       int(face.right()),int(face.bottom()))
    detectedLandmarks = faceLandmarkDetector(imgRGB, faceRectangleDlib)
    landmarks.append(detectedLandmarks)

    npyFileName = img_path.split('.')[0] + '.npy'

    if saveImg:
        imgFileName = img_path.split('.')[0] + '_lm.jpg'
        facePoints(img, detectedLandmarks)
        cv2.imwrite(imgFileName, img)        

    return writeFaceLandmarksToNPY(detectedLandmarks, npyFileName)

def detectLandmarkFast(img_path, frontalFaceDetector, faceLandmarkDetector,
                       saveNPY=True, saveImg=False):
    img = cv2.imread(img_path)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = frontalFaceDetector(imgRGB, 0)
    if len(face) == 0:
        tqdm.write(f'{img_path}: no face detected')
        return None

    face = face[0]
    landmarks = []

    faceRectangleDlib = dlib.rectangle(int(face.left()),int(face.top()),
                                       int(face.right()),int(face.bottom()))
    detectedLandmarks = faceLandmarkDetector(imgRGB, faceRectangleDlib)
    landmarks.append(detectedLandmarks)

    # npyFileName = img_path.split('.')[0] + '.npy'
    npyFileName = os.path.join('/home/t-yualee/data/oulu_npu/landmarks',
                               img_path.split('sample_processed/')[1]).split('.')[0] + '.npy'

    if saveImg:
        imgFileName = img_path.split('.')[0] + '_lm.jpg'
        facePoints(img, detectedLandmarks)
        cv2.imwrite(imgFileName, img)        

    return writeFaceLandmarksToNPY(detectedLandmarks, npyFileName)