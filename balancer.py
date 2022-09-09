import cv2
from ballTracking import preprocess, isolatePlate


video = cv2.VideoCapture(0)

while True:
    _, img = video.read()
    
    while img is None:
        pass    
        
    img = preprocess(img)
    img = isolatePlate(img)
    
    
    cv2.imshow("Isolated plate", img)
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows() 