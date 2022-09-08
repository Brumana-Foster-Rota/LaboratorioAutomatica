import cv2

from ballTracking import preprocess, isolatePlate, isolateBall


video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    
    while img is None:
        pass    
        
    img = preprocess(img)
    debugImg = img
    img = isolatePlate(img)
    ballX, ballY, debugImg = isolateBall(img, debug_img = debugImg)
    
    cv2.imshow("Debug", debugImg)
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows() 