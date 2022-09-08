import cv2
from time import sleep


video = cv2.VideoCapture(0)
sleep(2.0)

while True:
    _, frame = video.read()
    
    if frame is None:
        break
    
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    
video.release()
cv2.destroyAllWindows()    