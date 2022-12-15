---
layout: post
title: MediaPipe Hands poke bubbles
author: [蘇言文、陳柏帆]
category: [Lecture]
tags: [ai]
---


### MediaPipe Hands poke bubbles(戳泡泡遊戲)

**期末報告描述**

一個可以透過攝像頭遊玩的戳泡泡遊戲，透過MediaPipe模型辨識畫面中食指的位置，並戳破畫面中的泡泡來計分。

**開發步驟:**
1. <br>
2. <br>
3. <br>
4. <br>
5.  <br>

**執行環境:**
1. <br>
2. <br>
3. <br>
4. <br>
5.  <br>

### MediaPipe介紹


![](https://i.imgur.com/dtPZI4v.png)

![](https://i.imgur.com/MVqCJEP.png)


## 程式說明


程式的部分主要分成**辨識手掌**和**產生泡泡**還有**計分計時**三個部分，其中辨識手掌的部分主要參考自MediaPipe的官方文檔如下

### 辨識手掌
```python=


import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()



```

### 產生泡泡

```python=
if x>rx and x<(rx+25) and y>ry and y<(ry+25):
    run = True
```

```python=
rx = random.randint(50,w-50)
ry = random.randint(50,h-100)
```

```python=
cv2.circle(image, (rx, ry), 25, (255, 230, 190), -1)
cv2.circle(image, (rx-10, ry-10), 3, (250, 250, 250), 4)
cv2.circle(image, (rx + 10, ry + 10), 2, (250, 250, 250), 2)
cv2.circle(image, (rx, ry), 23, (230, 224, 176), 2)
```

### 計分計時


```python=
time_on = time.time()+30 
```

```python=
time_now = time.time()
time_li = str(time_on - time_now)
```

```python=
if time_on - time_now<= 0:
    cv2.rectangle(image, (130, 200),(550, 270),(0, 0, 0),-1)
    cv2.putText(image,"Your point is:" + str(fra) , (150,250), cv2.FONT_HERSHEY_DUPLEX, 1.5, (10, 215, 255), 2, cv2.LINE_AA)
```

```python=
cv2.rectangle(image, (270,50),(400,10),(192, 192, 192),-1)
cv2.rectangle(image, (250,460),(400,415),(192, 192, 192),-1)
cv2.putText(image,"Point:" + str(fra) , (270, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.putText(image,"Time:" + time_li[:4], (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2, cv2.LINE_AA)
```

**完整MediaPipe Hands poke bubbles(戳泡泡遊戲)程式** <br>

``` python
import cv2
import mediapipe as mp
import random
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

run=True
fra = -1
time_on = time.time()+30
cap = cv2.VideoCapture(0)



with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image,1)

    if not success:
      print("Ignoring empty camera frame.")
      continue

    image =cv2.resize(image,(640,480))
    size = image.shape
    w = size[1]
    h = size[0]

    if run:
        run = False
        fra = fra + 1
        rx = random.randint(50,w-50)
        ry = random.randint(50,h-100)

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        x = hand_landmarks.landmark[8].x * w
        y = hand_landmarks.landmark[8].y * h
        if x>rx and x<(rx+25) and y>ry and y<(ry+25):
            run = True
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    image2 = np.zeros((400, 400, 3), np.uint8)
    image2.fill(90)
    time_now = time.time()
    time_li = str(time_on - time_now)

    cv2.rectangle(image, (270,50),(400,10),(192, 192, 192),-1)
    cv2.rectangle(image, (250,460),(400,415),(192, 192, 192),-1)
    cv2.putText(image,"Point:" + str(fra) , (270, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image,"Time:" + time_li[:4], (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2, cv2.LINE_AA)

    cv2.circle(image, (rx, ry), 25, (255, 230, 190), -1)
    cv2.circle(image, (rx-10, ry-10), 3, (250, 250, 250), 4)
    cv2.circle(image, (rx + 10, ry + 10), 2, (250, 250, 250), 2)
    cv2.circle(image, (rx, ry), 23, (230, 224, 176), 2)
    if time_on - time_now<= 0:
          cv2.rectangle(image, (130, 200),(550, 270),(0, 0, 0),-1)
          cv2.putText(image,"Your point is:" + str(fra) , (150,250), cv2.FONT_HERSHEY_DUPLEX, 1.5, (10, 215, 255), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) == ord('q') or time_on - time_now<= 0:
      print("Your point is " + str(fra) )
      time.sleep(5)
      break

cap.release()
```



---
### 參考資料

https://google.github.io/mediapipe/

https://steam.oxxostudio.tw/category/python/ai/ai-mediapipe-hand.html#a1

https://www.kaggle.com/code/rkuo2000/mediapipe-pose


https://steam.oxxostudio.tw/category/python/ai/opencv-drawing.html#a4

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

