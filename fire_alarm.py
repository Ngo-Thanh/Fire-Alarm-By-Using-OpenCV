import cv2
import numpy as np
import pygame

pygame.mixer.init()

cap = cv2.VideoCapture(0)

lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])

alert_sound = "eas-alarm-phone-alarm-262882.wav"

alert_flag = False
cooldown_frames = 100
current_cooldown = cooldown_frames
min_contour_area = 1000

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        if not alert_flag and current_cooldown == cooldown_frames:
            pygame.mixer.music.load(alert_sound)
            pygame.mixer.music.play()
            print("Alert Detected")
            alert_flag = True
    if alert_flag:
        current_cooldown -= 1
        if current_cooldown == 0:
            alert_flag = False
            current_cooldown = cooldown_frames
    
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()