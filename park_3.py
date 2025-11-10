import cv2
import numpy as np
from keras.models import load_model

model = load_model("emptyparkingspotdetectionmodel.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

coordinates = [
    [(20, 8), (58, 88)],
    [(59, 8), (102, 87)],
    [(100, 4), (144, 85)],
    [(145, 8), (184, 87)],
    [(186, 10), (226, 89)],
    [(225, 9), (268, 90)],
    [(267, 9), (308, 88)],
    [(309, 8), (349, 89)],
    [(349, 7), (391, 89)],
    [(394, 9), (434, 90)],
    [(436, 10), (474, 91)],
    [(474, 9), (515, 91)],
    [(517, 12), (573, 94)],
    [(23, 194), (63, 276)],
    [(63, 194), (103, 278)],
    [(103, 197), (142, 277)],
    [(145, 196), (182, 274)],
    [(187, 197), (227, 278)],
    [(228, 198), (270, 275)],
    [(269, 190), (308, 275)],
    [(311, 199), (346, 272)],
    [(354, 196), (389, 272)],
    [(396, 196), (433, 273)],
    [(437, 195), (480, 275)],
    [(487, 201), (511, 273)],
    [(521, 199), (566, 271)],
    [(26, 282), (61, 361)],
    [(65, 284), (103, 359)],
    [(107, 281), (144, 362)],
    [(152, 287), (175, 365)],
    [(185, 281), (223, 363)],
    [(231, 284), (268, 359)],
    [(275, 287), (310, 362)],
    [(312, 284), (347, 361)],
    [(353, 284), (389, 363)],
    [(395, 284), (432, 365)],
    [(437, 285), (470, 364)],
    [(476, 282), (520, 370)],
    [(529, 290), (568, 361)]
]

def detect_empty_parking(image, spot):
    x1, y1 = spot[0]
    x2, y2 = spot[1]
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print("Invalid coordinates for ROI")
        return False
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        print("Empty ROI")
        return False
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, (48,48))
    resized_roi = resized_roi.astype('float32') / 255
    resized_roi = np.expand_dims(resized_roi, axis=0)
    resized_roi = np.expand_dims(resized_roi, axis=-1)
    prediction = model.predict(resized_roi)
    threshold = 0.01
    if prediction[0][0] > threshold:
        return True
    else:
        return False
    
current_image = cv2.imread("D:\\New folder (3)\\Park\\6ab56fe0604bf35a512a3731ddf26f28.jpg")
empty_count = 0

for spot in coordinates:
    if detect_empty_parking(current_image, spot):
        cv2.rectangle(current_image, spot[0], spot[1], (0,255,0), 2)
        empty_count += 1
    else: 
        cv2.rectangle(current_image, spot[0], spot[1], (0,0,255), 2)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(current_image, f"Empty Spots: {empty_count}", (50,50), font, 1.5, (255,255,255), 3, cv2.LINE_AA)

cv2.imshow("Parking Lot", current_image)
cv2.waitKey(0)
cv2.destroyAllWindows()









