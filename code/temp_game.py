import cv2
import mediapipe as mp
import random
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
fra = -1
time_on = time.time()+40



while cv2.waitKey(5) != ord('w'):
    time.sleep(1)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    run = True
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img,(640,480))
        size = img.shape
        w = size[1]
        h = size[0]
        if run:
            run = False
            fra = fra + 1
            rx = random.randint(50,w-50)
            ry = random.randint(50,h-100)
            print(rx, ry)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        results = hands.process(img2)                
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = hand_landmarks.landmark[7].x * w   
                y = hand_landmarks.landmark[7].y * h 
                print(x,y)
                if x>rx and x<(rx+80) and y>ry and y<(ry+80):
                    run = True
               
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        time_now = time.time()
        time_li = str(time_on - time_now)
        cv2.putText(img,"time:" + time_li[:4], (250,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img,"point:" + str(fra) , (270,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.circle(img, (rx, ry), 25, (255, 230, 190), -1)
        cv2.circle(img, (rx-10, ry-10), 3, (250, 250, 250), 4)
        cv2.circle(img, (rx + 10, ry + 10), 2, (250, 250, 250), 2)
        cv2.circle(img, (rx, ry), 23, (230, 224, 176), 2)
        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q') or time_on - time_now<= 0:
            print("Your point is " + str(fra) )
            break    
cap.release()
cv2.destroyAllWindows()