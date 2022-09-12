import cv2
import numpy as np
from ballTracking import preprocess, isolatePlate, isolateBall

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
from simple_pid import PID


# needs daemon pigpio to be running: on this machine it starts on startup
servoX = AngularServo(17, pin_factory = PiGPIOFactory(), min_angle = -90, max_angle = 90, max_pulse_width = 0.00255, min_pulse_width = 0.0004)
servoY = AngularServo(27, pin_factory = PiGPIOFactory(), min_angle = -90, max_angle = 90, max_pulse_width = 0.00245, min_pulse_width = 0.0004)


# Ziegler-Nichols + tests
PROP = 5.5 # 1.1 5.5 con altri valori default ha performato bene
Kp = 0.03 * .7 * PROP
Ki = 0.012 * 2.2 * PROP
Kd = 0.01875 * 3.5 * PROP

# setpoint yet to determine
controllerX = PID(Kp, Ki, Kd, sample_time = 0.0001)
controllerX.output_limits = (-90, 90)
controllerY = PID(Kp, Ki, Kd, sample_time = 0.0001)
controllerY.output_limits = (-90, 90)

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
#         controllerX.setpoint = int(targetX)
#         controllerY.setpoint = int(targetY)
        setup = False
    else:
        _, _, img = isolatePlate(img)
        
    ballX, ballY, img, imgDebug = isolateBall(img)
    
    # add control target to debug image
    imgDebug = cv2.line(imgDebug, (int(targetX) - 10, int(targetY)), (int(targetX) + 10, int(targetY)), (0, 255, 0), thickness = 1)
    imgDebug = cv2.line(imgDebug, (int(targetX), int(targetY) - 10), (int(targetX), int(targetY) + 10), (0, 255, 0), thickness = 1)   
    
    
    # balancing control
    if ballX is not None and ballY is not None:
#         ballX = kalmanX.getEstimate(ballX)[0]
#         ballY = kalmanY.getEstimate(ballY)[0]
        
        controlX = controllerX(ballX)
        controlY = controllerY(ballY)
#         controlX = controllerX(int(ballX))
#         controlY = controllerY(int(ballY))
        
        previousControlX.append(controlX)
        previousControlY.append(controlY)

        if len(previousControlX) >=5:
            controlX = np.mean(previousControlX[-4 : len(previousControlX)])
            controlY = np.mean(previousControlY[-4 : len(previousControlY)])

        servoX.angle = controlX
        servoY.angle = controlY
        
        # add control input to debug image
        imgDebug = cv2.putText(imgDebug, "setpointX - ballX: " + str(int(controllerX.setpoint) - int(ballX)) + " | setpointY - ballY: " + str(int(controllerY.setpoint) - int(ballY)), (int(imgDebug.shape[1] / 3), int(imgDebug.shape[0] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255))
        imgDebug = cv2.putText(imgDebug, "X: " + str(int(controlX)) + ", Y: " + str(int(controlY)), (int(imgDebug.shape[1] / 2), int(imgDebug.shape[0] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255))
 
    
    cv2.imshow("Debug", imgDebug)
    cv2.waitKey(1)


video.release()
cv2.destroyAllWindows()