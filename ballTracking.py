import cv2
import numpy as np
from time import sleep


def preprocess(img, scale = .5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # RGB => B/W
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale))) # X x Y => X * scale x Y * scale
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0) # blur image
    return img

def isolatePlate(img):
    # Highlight plate in white with thick black edges
    mask = cv2.Canny(img, 35, 35 / 3)
    kernel = np.ones((8, 8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    mask = np.bitwise_not(mask)
    # Get convex hull of the biggest white area
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    cnt = cnts[0]
    cnt = cv2.convexHull(cnt, False)
    # Set everything outside the plate as pure white
    mask = cv2.drawContours(mask * 0, [cnt], -1, 255, -1)
    img[mask != 255] = 255
    return img

def isolateBall(img, debug_img=None):
    # Find darkest pixels and erode them a bit to
    # cancel out any line or stick inside the image
    _, img = cv2.threshold(img, np.median(img) * .25, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    # Find average pixel coords, if there is any
    idxs = np.argwhere(img == 255)
    cx = cy = None
    if len(idxs) > 0:
        cy, cx = np.mean(idxs, axis=0)
        # Render image  with crosshair for debugging purposes
        if debug_img is not None:
            if len(debug_img.shape) == 2:
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
            ball_area = debug_img * 0
            debug_img[img == 255] = [255, 0,  0]
            # debug_img = cv2.addWeighted(debug_img, 1, ball_area, 1, 0)
            debug_img = cv2.line(debug_img, (int(cx), 0), (int(
                cx), debug_img.shape[0]), (255, 255, 255), thickness=2)
            debug_img = cv2.line(debug_img, (0, int(
                cy)), (debug_img.shape[1], int(cy)), (255, 255, 255), thickness=2)
        # cx /= img.shape[1]
        # cy /= img.shape[0]

    return cx, cy, debug_img


video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    
    if frame is None:
        break
    
    frame = preprocess(frame)
    preview = frame
    frame = isolatePlate(frame)
    cx, cy, preview = isolateBall(frame, debug_img = preview)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Preview", preview)
    cv2.waitKey(1)
    
video.release()
cv2.destroyAllWindows()    