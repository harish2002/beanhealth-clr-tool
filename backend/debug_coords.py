import sys
from PIL import Image
import numpy as np

from pipeline.module1_detection import detect_and_crop_eyes
from pipeline.module2_pupil import _map_landmarks_to_crop, _landmark_centre, _localise_one_eye

def run():
    img_path = "/Users/harish/Desktop/beanstrabis/backend/tests/test_images/flash_on_normal/Photo on 14-03-26 at 6.13 PM #3.jpg"
    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image: {e}")
        return

    img_rgb = np.array(pil_img, dtype=np.uint8)

    detection = detect_and_crop_eyes(img_rgb)
    print("----- RIGHT EYE -----")
    print("Crop Box: ", detection.right_crop_box)
    print("Landmarks Orig: ", detection.right_iris_landmarks)
    
    crops_mapped = _map_landmarks_to_crop(detection.right_iris_landmarks, detection.right_crop_box)
    print("Landmarks Crop Mapped: ", crops_mapped)
    
    lm_center = _landmark_centre(crops_mapped)
    print("Landmark Center (Crop Space): ", lm_center)
    from pipeline.module2_pupil import _preprocess_for_hough, _hough_estimate, _agree_and_fuse, _iris_radius_in_crop
    import cv2
    iris_r = _iris_radius_in_crop(crops_mapped)
    blurred = _preprocess_for_hough(detection.right_crop)
    
    cv2.imwrite("/tmp/hough_blurred_right.jpg", blurred)
    cv2.imwrite("/tmp/crop_right.jpg", detection.right_crop[..., ::-1])
    
    from utils.constants import HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1
    min_r = max(4, int(iris_r * 0.60))
    max_r = max(min_r + 4, int(iris_r * 1.40))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST, param1=HOUGH_PARAM1, param2=12, minRadius=min_r, maxRadius=max_r)
    
    debug_hough = cv2.cvtColor(detection.right_crop.copy(), cv2.COLOR_RGB2BGR)
    hough_res = None
    if circles is not None:
        best = circles[0][0]
        hough_res = (float(best[0]), float(best[1]), float(best[2]))
        for idx, c in enumerate(circles[0]):
            color = (0, 255, 0) if idx == 0 else (0, 0, 255)
            cv2.circle(debug_hough, (int(c[0]), int(c[1])), int(c[2]), color, 1)
            cv2.circle(debug_hough, (int(c[0]), int(c[1])), 2, color, -1)
            
    cv2.imwrite("/tmp/hough_debug_right.jpg", debug_hough)
    
    print("Iris Radius:", iris_r)
    print("Hough Result:", hough_res)
    
    flags = []
    fused, conf, hc, hr = _agree_and_fuse(lm_center, hough_res, "right", flags)
    print("Fused Center:", fused)
    print("Confidence:", conf)
    print("Flags:", flags)

    from pipeline.module3_clr import detect_clr
    from pipeline.module4_displacement import compute_displacement
    try:
        clr = detect_clr(detection.left_crop, detection.right_crop, iris_r, iris_r)
        print("CLR Right:", clr.right_clr)
        
        disp = compute_displacement((0,0), fused, (0,0), clr.right_clr, iris_r, iris_r)
        print("Displacement Norm Right:", disp.right_displacement_norm)
        print("Displacement DX:", disp.right_dx, "DY:", disp.right_dy)
        print("Magnitude:", disp.right_displacement_px)
    except Exception as e:
        print("CLR Failed:", e)

run()
