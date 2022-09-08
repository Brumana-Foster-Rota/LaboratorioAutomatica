import cv2
from ballTracking import preprocess


video = cv2.VideoCapture(0)

while True:
    _, img = video.read()
    
    while img is None:
        pass
    
    img = preprocess(img)
    edges = cv2.Canny(img, 125, 100)
    
    cv2.imshow("Image", img)
    cv2.imshow("Edges", edges)
    cv2.waitKey(1)
    
video.release()
cv2.destroyAllWindows()    