import cv2
import glob
import numpy as np


for file in glob.glob("Dataset/*.bmp"):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 30, 200)

    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    contours = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    max_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [max_contour], 0, (255, 255, 255), cv2.FILLED)

    mask_inv = 255 - mask

    image_masked = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('Image', image_masked)

    key = cv2.waitKey(0)
    if key == 2555904:
        continue
    elif key == 113:
        break
