import cv2
from ballTracking import preprocess, isolatePlate


video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    
    while img is None:
        pass    
        
    img = preprocess(img)
    img = isolatePlate(img)

video.release()
cv2.destroyAllWindows() 