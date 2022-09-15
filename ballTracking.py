import cv2
import numpy as np


# makes the image more suitable for object detection by converting it to grayscale and resizing it (smaller image allows smaller kernel and faster computing)
def preprocess(img, scale = 0.5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # RGB => B/W
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale))) # X x Y => X * scale x Y * scale
    
    return img


# makes everything in img that is not the plate white, leaving the plate visible and returning the coordinates of its center
def isolatePlate(img):
    edges = cv2.Canny(img, 125, 100) # Canny detection of edges, thresholds for hysteresis thresholding determined experimentally
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, np.ones((8, 8))) # morphological dilation of pixels near edges [edges => white]
    inverseEdges = np.bitwise_not(edges) # [edges => black]

    contours, _ = cv2.findContours(inverseEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finds contours of white shapes in inverseEdges
    contours = sorted(contours, key = cv2.contourArea, reverse = True) # orders contours by area they include
    plateContour = contours[0] # plate's contour will have the biggest area among contours (if the setup is correct)
    
    plate = cv2.convexHull(plateContour, False) # finds precise plate shape
    plateMask = cv2.drawContours(inverseEdges * 0, [plate], -1, 255, -1) # draws plate on pitch black background same size as img
    platePoints = np.argwhere(plateMask == 255) # lists coordinates of points plate occupies
    
    plateCenterY, plateCenterX = np.mean(platePoints, axis = 0)
    img[plateMask != 255] = 255 # everything that is not plate is pitch white (plateMask over img and all that's 0 becomes 255)
    return plateCenterX, plateCenterY, img


# key for sorting list of detected balls in isolateBall
def ballRadius(ball):
     return ball[0][2]

# isolates the ball from the plate and returns its coordinates both in numerical and graphical form for debugging 
def isolateBall(img):  
    ballX = ballY = None
    imgDebug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # recommended parameters for ball detection + radius limitation for excluding fake positives
    balls = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 16)
    if balls is not None:        
        balls = sorted(balls, key = ballRadius, reverse = True) # first ball in list => biggest ball
        ballX = balls[0][0][0]
        ballY = balls[0][0][1]
        # add coordinates to RGB image for debugging (above ball)
        imgDebug = cv2.putText(imgDebug, "(" + str(round(ballX, 2)) + ", " + str(round(ballY, 2)) + ")", (int(ballX), int(ballY)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0))

    return ballX, ballY, img, imgDebug

# isolates the ball from the plate and returns its coordinates both in numerical and graphical form for debugging 
def isolateBallV1(img):
    _, thresholdedImg = cv2.threshold(img, np.median(img) * 0.1, 255, cv2.THRESH_BINARY_INV) # inverse thresholding of img, ball will come out white
    thresholdedImg = cv2.morphologyEx(thresholdedImg, cv2.MORPH_ERODE, np.ones((3, 3))) # morphological erosion of black pixels to remove possible black spots
    
    # median coordinates of white pixels (only the ball will be white at this point) [img matrix row => plate Y]
    ballArea = np.argwhere(thresholdedImg == 255)   
    ballX = ballY = None
    imgDebug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(ballArea) > 0:
        ballY, ballX = np.mean(ballArea, axis = 0)
        # add coordinates to RGB image for debugging (above ball)
        imgDebug = cv2.putText(imgDebug, "(" + str(round(ballX, 2)) + ", " + str(round(ballY, 2)) + ")", (int(ballX), int(ballY)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0))
    
    return ballX, ballY, img, imgDebug