import glob
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--directory", default="Dataset", help="Directory containing images and template image")
args = parser.parse_args()

# out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 2, (1024, 1024))

template = cv2.imread(args.directory + "\\template.png", cv2.IMREAD_GRAYSCALE)
assert template is not None
w, h = template.shape[::-1]

angle = 0.0
prev_angle = None

orb = cv2.ORB_create()

for i, file in enumerate(glob.glob(args.directory + "/*.bmp")):
    if i == 25:
        template = cv2.imread(args.directory + "\\template2.png", cv2.IMREAD_GRAYSCALE)
        assert template is not None
        w, h = template.shape[::-1]

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    img_kp, img_des = orb.detectAndCompute(img, None)
    template_kp, template_des = orb.detectAndCompute(template, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(template_des, img_des)
    matches = sorted(matches, key=lambda x: x.distance)

    template_pts = np.float32([template_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_pts = np.float32([img_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(template_pts, img_pts, cv2.RANSAC)

    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M)

    rect = cv2.minAreaRect(transformed_corners)
    box = cv2.boxPoints(rect).astype(int)

    if prev_angle is None:
        prev_angle = rect[2]
    elif rect[2] < prev_angle and abs(rect[2] - prev_angle) > 50:
        angle = (angle + 90 - prev_angle + rect[2]) % 360
        prev_angle = rect[2]
    else:
        angle = (angle + rect[2] - prev_angle) % 360
        prev_angle = rect[2]

    cv2.putText(img, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 2)
    cv2.drawContours(img, [box], 0, 255, 2)

    # out.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    cv2.imshow("Image", img)

    key = cv2.waitKey(0)
    if key == 2555904:
        continue
    elif key == 113:
        break

# out.release()
