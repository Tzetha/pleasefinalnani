import cv2

# Test camera 0
cap = cv2.VideoCapture(6)
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()