import cv2
import numpy as np
from ballTracking import preprocess, isolatePlate, isolateBall


video = cv2.VideoCapture(0)

while True:
    _, img = video.read()
    
    while img is None:
        pass
    
    img = preprocess(img)
    img = isolatePlate(img)
    ballX, ballY, img, imgDebug = isolateBall(img)


    cv2.imshow("Processed image", img)
    cv2.imshow("Debug image", imgDebug)
    
    cv2.waitKey(1)
    
    
video.release()
cv2.destroyAllWindows()    