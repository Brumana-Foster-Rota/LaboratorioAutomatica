import cv2
import numpy as np
from time import sleep

# makes the image more suitable for object detection by converting it to grayscale and resizing it (smaller image allows smaller kernel and faster computing)
def preprocess(img, scale = 0.5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # RGB => B/W
    cv2.imwrite('preprocess1.jpg', img)
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale))) # X x Y => X * scale x Y * scale
    cv2.imwrite('preprocess2.jpg', img)
    return img


# makes everything in img that is not the plate white, leaving the plate visible and returning the coordinates of its center
def isolatePlate(img):
    edges = cv2.Canny(img, 125, 100) # Canny detection of edges, thresholds for hysteresis thresholding determined experimentally
    cv2.imwrite('isolatePlate1.jpg', edges)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, np.ones((8, 8))) # morphological dilation of pixels near edges [edges => white]
    cv2.imwrite('isolatePlate2.jpg', edges)
    inverseEdges = np.bitwise_not(edges) # [edges => black]
    cv2.imwrite('isolatePlate3.jpg', inverseEdges)

    contours, _ = cv2.findContours(inverseEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finds contours of white shapes in inverseEdges
    contours = sorted(contours, key = cv2.contourArea, reverse = True) # orders contours by area they include
    plateContour = contours[0] # plate's contour will have the biggest area among contours (if the setup is correct)
    
    plate = cv2.convexHull(plateContour, False) # finds precise plate shape
    plateMask = cv2.drawContours(inverseEdges * 0, [plate], -1, 255, -1) # draws plate on pitch black background same size as img
    cv2.imwrite('isolatePlate4.jpg', plateMask)
    platePoints = np.argwhere(plateMask == 255) # lists coordinates of points plate occupies
    
    plateCenterY, plateCenterX = np.mean(platePoints, axis = 0)
    img[plateMask != 255] = 255 # everything that is not plate is pitch white (plateMask over img and all that's 0 becomes 255)
    cv2.imwrite('isolatePlate5.jpg', img)
    return plateCenterX, plateCenterY, img


# isolates the ball from the plate and returns its coordinates both in numerical and graphical form for debugging 
def isolateBall(img):
    _, thresholdedImg = cv2.threshold(img, np.median(img) * .4, 255, cv2.THRESH_BINARY_INV) # inverse thresholding of img, ball will come out white
    cv2.imwrite('isolateBall1.jpg', thresholdedImg)
    thresholdedImg = cv2.morphologyEx(thresholdedImg, cv2.MORPH_ERODE, np.ones((3, 3))) # morphological erosion of black pixels to remove possible black spots
    cv2.imwrite('isolateBall2.jpg', thresholdedImg)
    
    # median coordinates of white pixels (only the ball will be white at this point) [img matrix row => plate Y]
    ballArea = np.argwhere(thresholdedImg == 255)
    ballX = ballY = None
    imgDebug = img
    if len(ballArea) > 0:
        ballY, ballX = np.mean(ballArea, axis = 0)
        # add coordinates to RGB image for debugging (above ball)
        imgDebug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imgDebug = cv2.putText(imgDebug, "(" + str(round(ballX, 2)) + ", " + str(round(ballY, 2)) + ")", (int(ballX), int(ballY)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0))

    return ballX, ballY, img, imgDebug


video = cv2.VideoCapture(0)
sleep(1)

_, img = video.read()

img = preprocess(img)
_, _, img = isolatePlate(img)
_, _, img, _ = isolateBall(img)