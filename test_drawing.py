import cv2
import numpy as np

# Simulate a brown squiggle on black canvas
w, h = 300, 300
img = np.zeros((h, w, 3), dtype=np.uint8)
cv2.line(img, (50, 50), (250, 250), (42, 42, 165), 20)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Unique gray values ( squiggle ):", len(np.unique(gray)))

# Simulate a real photo
photo = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
# smoothen it slightly to be more like a photo
photo = cv2.GaussianBlur(photo, (5, 5), 0)
photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

print("Unique gray values ( real photo ):", len(np.unique(photo_gray)))
