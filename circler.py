import cv2
import numpy as np
from ballTracking import preprocess, isolatePlate, isolateBall

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
from simple_pid import PID
from math import pi, cos, sin


# needs daemon pigpio to be running: on this machine it starts on startup
servoX = AngularServo(17, pin_factory = PiGPIOFactory(), min_angle = -90, max_angle = 90, max_pulse_width = 0.00255, min_pulse_width = 0.0004)
servoY = AngularServo(27, pin_factory = PiGPIOFactory(), min_angle = -90, max_angle = 90, max_pulse_width = 0.00245, min_pulse_width = 0.0004)


# determined via tests
Kp = 0.6 * 1.5
Ki = 0#0.375 * 1.1
Kd = 0.3 * 1.5

# setpoint yet to determine
controllerX = PID(Kp, Ki, Kd, sample_time = 0.0001)
controllerX.output_limits = (-45, 45)
controllerY = PID(Kp, Ki, Kd, sample_time = 0.0001)
controllerY.output_limits = (-45, 45)

angleDeg = 0
circleRadius = 60

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
    
    img = preprocess(img)
    plateCenterX, plateCenterY, img = isolatePlate(img)
    ballX, ballY, img, imgDebug = isolateBall(img)
    
    
    angleDeg = (angleDeg + 1) % 360
    angleRad = angleDeg * pi / 180
    targetX = plateCenterX + circleRadius * cos(angleRad)
    targetY = plateCenterY + circleRadius * sin(angleRad)
    
    controllerX.output_limits = (-45 * targetX / plateCenterX, 45 * plateCenterX / targetX)
    controllerY.output_limits = (-45 * targetY / plateCenterY, 45 * plateCenterY / targetY)
    
    controllerX.setpoint = targetX
    controllerY.setpoint = targetY
    
    
    # add control target to debug image
    imgDebug = cv2.line(imgDebug, (int(targetX) - 10, int(targetY)), (int(targetX) + 10, int(targetY)), (0, 255, 0), thickness = 1)
    imgDebug = cv2.line(imgDebug, (int(targetX), int(targetY) - 10), (int(targetX), int(targetY) + 10), (0, 255, 0), thickness = 1)   
    
    
    # circling control
    if ballX is None:
        if len(previousBallX) >= 1:
            ballX = previousBallX[len(previousBallX) - 1]
        else:
            ballX = 135
        
    if ballY is None:
        if len(previousBallY) >= 1:
            ballY = previousBallY[len(previousBallY) - 1]
        else:
            ballY = 135
        
    previousBallX.append(ballX)
    previousBallY.append(ballY)

    controlX = controllerX(ballX)
    controlY = controllerY(ballY)
    
    previousControlX.append(controlX)
    previousControlY.append(controlY)
        
    # "low pass" filter that tries to reduce the impact of wrong ball detections (they sometimes happen)
    if len(previousControlX) >= 2 and (abs(previousControlX[len(previousControlX) - 1] - previousControlX[len(previousControlX) - 2]) > 22.5 or abs(previousControlY[len(previousControlY) - 1] - previousControlY[len(previousControlY) - 2]) > 22.5):
        controlX = np.mean(previousControlX[-2 : len(previousControlX)])
        controlY = np.mean(previousControlY[-2 : len(previousControlY)])

    servoX.angle = controlX
    servoY.angle = controlY
        
    # add control input to debug image
    imgDebug = cv2.putText(imgDebug, "setpointX - ballX: " + str(round(controllerX.setpoint - ballX, 2)) + " | setpointY - ballY: " + str(round(controllerY.setpoint - ballY, 2)), (int(imgDebug.shape[1] / 3), int(imgDebug.shape[0] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255))
    imgDebug = cv2.putText(imgDebug, "X: " + str(round(controlX, 2)) + ", Y: " + str(round(controlY, 2)), (int(imgDebug.shape[1] / 2), int(imgDebug.shape[0] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255))

    
    cv2.imshow("Debug", imgDebug)
    cv2.waitKey(1)


video.release()
cv2.destroyAllWindows()