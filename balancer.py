import cv2
import numpy as np
from ballTracking import preprocess, isolatePlate, isolateBall

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
from simple_pid import PID


# needs daemon pigpio to be running: on this machine it starts on startup
servoX = AngularServo(17, pin_factory = PiGPIOFactory(), min_angle = -90, max_angle = 90, max_pulse_width = 0.00255, min_pulse_width = 0.0004)
servoY = AngularServo(27, pin_factory = PiGPIOFactory(), min_angle = -90, max_angle = 90, max_pulse_width = 0.00245, min_pulse_width = 0.0004)


# determined via tests
Kp = 0.65
Ki = 0 #0.15 
Kd = 0.19#0.28

# setpoint yet to determine
controllerX = PID(Kp, Ki, Kd, sample_time = 0.0001)
controllerX.output_limits = (-45, 45)
controllerY = PID(Kp, Ki, Kd, sample_time = 0.0001)
controllerY.output_limits = (-45, 45)

previousBallX = []
previousBallY = []
previousControlX = []
previousControlY = []


setup = True
video = cv2.VideoCapture(0)

input("Place a ball on the plate and press a key to start the balancing control:")
while True:
    _, img = video.read()
    
    while img is None:
        pass    
    
    # detection of ball and center of plate if first execution (plate will be perfectly level on that occasion)
    img = preprocess(img)
    
    if setup:
        targetX, targetY, img = isolatePlate(img) # recalculating center every time generates very variable setpoint which makes control noisy
        controllerX.setpoint = targetX
        controllerY.setpoint = targetY
        setup = False
    else:
        _, _, img = isolatePlate(img)
        
    ballX, ballY, img, imgDebug = isolateBall(img, plateMask)
    
    # add control target to debug image
    imgDebug = cv2.line(imgDebug, (int(targetX) - 10, int(targetY)), (int(targetX) + 10, int(targetY)), (0, 255, 0), thickness = 1)
    imgDebug = cv2.line(imgDebug, (int(targetX), int(targetY) - 10), (int(targetX), int(targetY) + 10), (0, 255, 0), thickness = 1)   
    
    
    # balancing control
    if ballX is not None and ballY is not None:
        previousBallX.append(ballX)
        previousBallY.append(ballY)
        
        controlX = controllerX(ballX)
        controlY = controllerY(ballY)

        previousControlX.append(controlX)
        previousControlY.append(controlY)
        
#         if len(previousControlX) >= 3:
#             controlX = np.mean(previousControlX[-3 : len(previousControlX)])
#             controlY = np.mean(previousControlY[-3 : len(previousControlY)])
        
        servoX.angle = controlX
        servoY.angle = controlY
        
        # add control input to debug image
        imgDebug = cv2.putText(imgDebug, "setpointX - ballX: " + str(round(controllerX.setpoint - ballX, 2)) + " | setpointY - ballY: " + str(round(controllerY.setpoint - ballY, 2)), (int(imgDebug.shape[1] / 3), int(imgDebug.shape[0] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255))
        imgDebug = cv2.putText(imgDebug, "X: " + str(round(controlX, 2)) + ", Y: " + str(round(controlY, 2)), (int(imgDebug.shape[1] / 2), int(imgDebug.shape[0] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255))

    
    cv2.imshow("Debug", imgDebug)
    cv2.imshow("Edges", edges)
    cv2.waitKey(1)


video.release()
cv2.destroyAllWindows()