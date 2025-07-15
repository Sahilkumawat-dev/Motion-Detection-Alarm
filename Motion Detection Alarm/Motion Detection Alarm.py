import cv2
import time
import numpy as np
import pyttsx3  # For text-to-speech

# Initialize the camera
cap = cv2.VideoCapture(0)  #default webcam

fgbg = cv2.createBackgroundSubtractorMOG2()

engine = pyttsx3.init()

# Function to play voice alarm
def play_voice_alarm(message="Motion detected!"):
    engine.say(message)
    engine.runAndWait()

# Variables for detecting motion
motion_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

    # If motion is detected, play the voice alarm
    if motion_detected:
        play_voice_alarm("Motion detected!")  # Play voice message
        motion_detected = False

    cv2.imshow("Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
