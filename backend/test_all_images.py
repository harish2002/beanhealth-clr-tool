import os
import glob
from PIL import Image
import numpy as np

from pipeline.module1_detection import detect_and_crop_eyes
from pipeline.module2_pupil import localise_pupils
from pipeline.module3_clr import detect_clr
from pipeline.module4_displacement import compute_displacement
from pipeline.module5_asymmetry import compute_asymmetry_and_angle
from pipeline.module6_classify import classify_strabismus
from utils.exceptions import CLRPipelineError
import cv2

def run():
    base_dir = "/Users/harish/Desktop/beanstrabis/backend/tests/test_images"
    images = glob.glob(os.path.join(base_dir, "**/*.*"), recursive=True)
    images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for idx, img_path in enumerate(images):
        print(f"--- Processing {os.path.basename(img_path)} ---")
        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_rgb = np.array(pil_img, dtype=np.uint8)
            
            det = detect_and_crop_eyes(img_rgb)
            pupil = localise_pupils(det.left_crop, det.right_crop, det.left_iris_landmarks, det.right_iris_landmarks, det.left_crop_box, det.right_crop_box)
            clr = detect_clr(det.left_crop, det.right_crop, pupil.left_iris_radius, pupil.right_iris_radius, pupil.left_pupil, pupil.right_pupil)
            disp = compute_displacement(
                pupil.left_pupil, pupil.right_pupil,
                clr.left_clr, clr.right_clr,
                pupil.left_iris_radius, pupil.right_iris_radius
            )
            asym = compute_asymmetry_and_angle(disp.left_displacement_norm, disp.right_displacement_norm, disp.flags)
            dominant_dir = disp.left_direction if asym.dominant_eye != "right" else disp.right_direction
            out = classify_strabismus(
                dominant_direction=dominant_dir,
                severity=asym.severity,
                asymmetry_score=asym.asymmetry_score,
                upstream_flags=asym.flags,
            )
            
            print(f"Result: {out.urgency_tier} - {out.condition_name} ({out.icd10_code})")
            print(f"Left Pupil: {pupil.left_pupil}, Right Pupil: {pupil.right_pupil}")
            print(f"Left CLR: {clr.left_clr}, Right CLR: {clr.right_clr}")
            
            # Draw it
            test_img = det.right_crop.copy()[..., ::-1] # BGR
            cv2.circle(test_img, (int(pupil.right_pupil[0]), int(pupil.right_pupil[1])), 4, (20, 100, 235), -1)
            cv2.circle(test_img, (int(clr.right_clr[0]), int(clr.right_clr[1])), 4, (255, 180, 20), -1)
            cv2.imwrite(f"/tmp/test_out_{idx}.jpg", test_img)
            
        except CLRPipelineError as e:
            print(f"Pipeline Error: {e.code}")
        except Exception as e:
            print(f"System Error: {e}")

run()
