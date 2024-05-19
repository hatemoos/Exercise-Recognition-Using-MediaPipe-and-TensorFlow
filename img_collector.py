import os
import cv2

DATA_DIR = './data_lunges'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

categories = ['correct', 'incorrect']
dataset_size = 600

cap = cv2.VideoCapture(0)
for category in categories:
    if not os.path.exists(os.path.join(DATA_DIR, category)):
        os.makedirs(os.path.join(DATA_DIR, category))

    print(f'Collecting data for category: {category}')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, category, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
