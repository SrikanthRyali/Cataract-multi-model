import os
import cv2
import numpy as np
from gradio_app import is_eye_image

with open("validation_results.txt", "w") as f:
    noise = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    cv2.imwrite("test_noise.png", noise)
    valid, msg = is_eye_image("test_noise.png")
    f.write(f"Noise image result: {valid}, {msg}\n")

    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.imwrite("test_blank.png", blank)
    valid, msg = is_eye_image("test_blank.png")
    f.write(f"Blank image result: {valid}, {msg}\n")
    
    from gradio_app import MAX_LAP_VARIANCE
    f.write(f"MAX_LAP_VARIANCE: {MAX_LAP_VARIANCE}\n")
