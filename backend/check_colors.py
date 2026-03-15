import cv2
import numpy as np

img = cv2.imread("/tmp/crop_right.jpg")
if img is None:
    print("Cannot read image")
else:
    # pupil is supposed to be at 128, 59
    # clr is at 160, 80
    
    # Let's get a 5x5 patch around each
    p_center = img[57:62, 126:131]
    clr_center = img[78:83, 158:163]
    
    print("Average color at MediaPipe Pupil (128, 59) [BGR]:", np.mean(p_center, axis=(0,1)))
    print("Average color at CLR (160, 80) [BGR]:", np.mean(clr_center, axis=(0,1)))
