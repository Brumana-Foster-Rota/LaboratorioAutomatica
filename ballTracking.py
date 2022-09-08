import cv2
import numpy as np
from time import sleep


# makes the image more suitable for object detection by converting it to grayscale and resizing it (smaller image allows smaller kernel and faster computing)
def preprocess(img, scale = 0.5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # RGB => B/W
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale))) # X x Y => X * scale x Y * scale
    return img

# highlights everything in img that is not the plate in white, leaving the plate visible (edge detection, matrix manipulation)
def isolatePlate(img):
    edges = cv2.Canny(img, 125, 100) # Canny detection of edges, thresholds for hysteresis thresholding determined experimentally
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, np.ones((8, 8))) # morphological dilation of pixels near edges [edges => white]
    inverseEdges = np.bitwise_not(edges) # [edges => black]
    
    contours, _ = cv2.findContours(inverseEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finds contours of white shapes in inverseEdges
    contours = sorted(contours, key = cv2.contourArea, reverse = True) # orders contours by area they include
    plateContour = contours[0] # plate's contour will be be the biggest one among contours (if the setup is correct)
    
    plate = cv2.convexHull(plateContour, False) # finds precise plate shape
    plateMask = cv2.drawContours(inverseEdges * 0, [plate], -1, 255, -1) # draws plate on pitch black background same size as img
    img[plateMask != 255] = 255 # everything that is not plate is pitch black (plateMask over img and all that's 0 becomes 255)
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