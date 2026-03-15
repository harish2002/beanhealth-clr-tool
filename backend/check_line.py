import cv2
import numpy as np

img = cv2.imread("/tmp/crop_right.jpg", cv2.IMREAD_GRAYSCALE)
if img is not None:
    y = 59
    line = img[y, :] # shape is (249,)
    
    # Let's print the pixel intensity every 5 pixels
    print("Intensity at Y=59:")
    for x in range(0, img.shape[1], 10):
        print(f"X={x:3d}: {line[x]:3d}")
    
    y = 80
    line2 = img[y, :]
    print("\nIntensity at Y=80 (where CLR is):")
    for x in range(0, img.shape[1], 10):
        print(f"X={x:3d}: {line2[x]:3d}")
