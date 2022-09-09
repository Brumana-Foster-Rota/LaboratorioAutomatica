import cv2
import numpy as np
from ballTracking import preprocess, isolatePlate, isolateBall

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
from simple_pid import PID

# needs daemon pigpio to be running: on this machine it starts on startup
servoX = AngularServo(17, pin_factory = PiGPIOFactory(), min_angle = 90, max_angle = -90, max_pulse_width = 0.00255, min_pulse_width = 0.0004)
servoY = AngularServo(27, pin_factory = PiGPIOFactory(), min_angle = 90, max_angle = -90, max_pulse_width = 0.00255, min_pulse_width = 0.0004)


PROP = 1.1
Kp = 0.03 * .7 * PROP
Ki = 0.012 * 2.2 * PROP
Kd = 0.01875 * 3.5 * PROP

controllerX = PID(Kp, Ki, Kd, sample_time = 0.0001)
controllerX.output_limits = (-20, 20)
controllerY = PID(Kp, Ki, Kd, sample_time = 0.0001)
controllerY.output_limits = (-20, 20)


video = cv2.VideoCapture(0)

while True:
    _, img = video.read()
    
    while img is None:
        pass    
    
    # ball and center of plate detection 
    img = preprocess(img)
    targetX, targetY, img = isolatePlate(img)
    ballX, ballY, img, imgDebug = isolateBall(img)
    
    # add control target to debug image
    sz = int(max(imgDebug.shape) * 0.1)
    imgDebug = cv2.line(imgDebug, (int(targetX) - sz, int(targetY)), (int(targetX) + sz, int(targetY)), (0, 255, 0), thickness = 2)
    imgDebug = cv2.line(imgDebug, (int(targetX), int(targetY) - sz), (int(targetX), int(targetY) + sz), (0, 255, 0), thickness = 2)   
    
    if ballX is not None and ballY is not None:
        controllerX.setpoint = targetX
        controllerY.setpoint = targetY

        controlX = controllerX(ballX)
        controlY = controllerY(ballY)

        servoX.angle = controlX
        servoY.angle = controlY
    
    cv2.imshow("Debug", imgDebug)
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows() 