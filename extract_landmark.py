import os
import os.path
from landmark import landmark
from tqdm import tqdm, trange

import dlib, cv2

frontalFaceDetector = dlib.get_frontal_face_detector()
faceLandmarkDetector = dlib.shape_predictor('landmark/shape_predictor_68_face_landmarks.dat')

root_dir = '/home/t-yualee/data/oulu_npu/sample_processed/cropped'

for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
    for filename in [f for f in filenames if f.endswith('.png')]:
        _ = landmark.detectLandmarkFast(os.path.join(dirpath, filename), frontalFaceDetector, faceLandmarkDetector)