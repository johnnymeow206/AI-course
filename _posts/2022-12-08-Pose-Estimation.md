---
layout: post
title: Pose Estimation
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Pose Estimation includes Applications, Body Pose, Head Pose, Hand Pose , Object Pose.


---
## Hand Pose
[Hand Pose Estimation papers](https://codechina.csdn.net/mirrors/xinghaochen/awesome-hand-pose-estimation)

### Hand3D
**Paper:** [arxiv.org/abs/1705.01389](https://arxiv.org/abs/1705.01389)<br>
**Code:** [lmb-freiburg/hand3d](https://github.com/lmb-freiburg/hand3d)<br>
![](https://github.com/lmb-freiburg/hand3d/blob/master/teaser.png?raw=true)

### DeeHPS
**Paper:** [arxiv.org/abs/1808.09208](https://arxiv.org/abs/1808.09208)<br>

![](https://www.researchgate.net/profile/Alexis-Heloir/publication/328310189/figure/fig1/AS:692903412240387@1542212455475/a-An-overview-of-our-method-for-simultaneous-3D-hand-pose-and-surface-estimation-A.ppm)

### GraphPoseGAN
**Paper:** [arxiv.org/abs/1912.01875](https://arxiv.org/abs/1912.01875)<br>

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/9479a278f3c63b57b26eb31e44744a238b11feee/1-Figure1-1.png)

### 3D Hand Shape
**Paper:** [arxiv.org/abs/1903.00812](https://arxiv.org/abs/1903.00812)<br>
**Code:** [https://github.com/3d-hand-shape/hand-graph-cnn](https://github.com/3d-hand-shape/hand-graph-cnn)<br>

![](https://github.com/3d-hand-shape/hand-graph-cnn/blob/master/teaser.png?raw=true)

---
### FaceBook InterHand2.6M
**Paper:** [InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image
](https://arxiv.org/abs/2008.09309)<br>
**Code:** [facebookresearch/InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M)<br>

![](https://github.com/facebookresearch/InterHand2.6M/blob/main/assets/teaser.gif?raw=true)

---


---
### Pose Recognition (姿態辨識)

**專題實作步驟:**
1. 建立身體動作之姿態照片資料集 (例如：5 poses , take 20 pictures of each pose)<br>
2. 始用**MMPose** 辨識出照片中的各姿勢之身體關鍵點 (use MMPose convert 16 keypoints (x,y) of each pose)<br>
3. 產生姿態關鍵點資料集 x_train.append(pose_keypoints) ( x_train.shape = (20x5, 16, 2), y_train.shape= (20x5, 1) )<br>
4. 建立DNN模型並訓練模型, 然後下載模型檔`pose_dnn.h5`至PC <br>
5. 於PC建立帶camera輸入之服務器程式, 載入模型`pose_dnn.h5`進行姿態動作辨識 <br>

**模型建構與訓練之程式樣本** (PC or Kaggle)<br>

```
input_shape=(16,2)
num_classes=5

inputs = layers.Input(shape=input_shape)
x = layers.Dense(128)(inputs)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs=inputs, outputs=outputs)

models.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size=1, epochs=20, validation_data=(x_test, y_test))
models.save_model(model, 'pose_dnn.h5')
```

**姿態辨識服務器之程式樣本** (PC with Camera)<br>

```
model = models.load_model('models/pose_dnn.h5')
labels = ['stand', 'raise-right-arm', 'raise-left-arm', 'cross arms','both-arms-left']

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    mmdet_results = inference_detector(det_model, image) # 人物偵測產生BBox
    person_results = process_mmdet_results(mmdet_results, args.det_cat_id) # 記住人物之BBox  
    pose_results, returned_outputs = inference_top_down_pose_model(...) # 感測姿態產生pose keypoints
    
    x_test = np.array(preson_results).reshape(1,16,2) # 將Keypoints List 轉成 numpy Array
    preds = model.fit(x_test) # 辨識姿態動作
    maxindex = int(np.argmax(preds))
    txt = labels[maxindex]
    print(txt)
```

---



<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

