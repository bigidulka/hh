import cv2
import numpy as np

PHOTO_FACE = 'face1.jpg'
SUNGLASSES =  'sunglasses2.png'
BLUR = True
DRAW_SUNGLASSES = True
DRAW_CIRCLES = True

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread(PHOTO_FACE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sunglasses_img = cv2.imread(SUNGLASSES, cv2.IMREAD_UNCHANGED)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    roi_color = img[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    face_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    eyes = eye_cascade.detectMultiScale(roi_gray)

    if DRAW_CIRCLES:
        cv2.circle(img, (x + w // 2, y + h // 2), max(w, h) // 2, (0, 255, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            eye_center = (ex + ew // 2, ey + eh // 2)
            eye_radius = max(ew, eh) // 2
            cv2.circle(img, (x + eye_center[0], y + eye_center[1]), eye_radius, (0, 0, 255), 2)
    
    if BLUR:
        eyes = eye_cascade.detectMultiScale(roi_gray)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_axes = (w // 2, h // 2)
        cv2.ellipse(face_mask, (w // 2, h // 2), face_axes, 0, 0, 360, 255, -1)
        
        for (ex, ey, ew, eh) in eyes:
            eye_center = (ex + ew // 2, ey + eh // 2)
            eye_radius = max(ew, eh) // 2
            cv2.circle(face_mask, eye_center, eye_radius, 0, -1)
        
        blurred_face = cv2.GaussianBlur(roi_color, (0, 0), 30)
        roi_color[face_mask == 255] = blurred_face[face_mask == 255]

    if DRAW_SUNGLASSES and len(eyes) == 2:
        eye_x1 = min(eyes[0][0], eyes[1][0])
        eye_x2 = max(eyes[0][0] + eyes[0][2], eyes[1][0] + eyes[1][2])
        eye_width = eye_x2 - eye_x1
        sunglasses_width = eye_width * 2
        sunglasses_height = int(sunglasses_img.shape[0] * sunglasses_width / sunglasses_img.shape[1])

        new_sunglasses = cv2.resize(sunglasses_img, (sunglasses_width, sunglasses_height), interpolation=cv2.INTER_AREA)

        sg_x1 = x + eye_x1 - eye_width // 4 - 10  # по верт
        sg_y1 = y + eyes[0][1] - sunglasses_height // 5 - 50  # по диаг
        sg_x2 = sg_x1 + sunglasses_width
        sg_y2 = sg_y1 + sunglasses_height

        sg_x1 = max(sg_x1, x)
        sg_y1 = max(sg_y1, y)
        sg_x2 = min(sg_x2, x + w)
        sg_y2 = min(sg_y2, y + h)

        sunglasses_width = sg_x2 - sg_x1
        sunglasses_height = sg_y2 - sg_y1
        new_sunglasses = cv2.resize(sunglasses_img, (sunglasses_width, sunglasses_height), interpolation=cv2.INTER_AREA)

        alpha_s = new_sunglasses[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            roi_color[sg_y1-y:sg_y2-y, sg_x1-x:sg_x2-x, c] = (alpha_s * new_sunglasses[:, :, c] + alpha_l * roi_color[sg_y1-y:sg_y2-y, sg_x1-x:sg_x2-x, c])
    
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
