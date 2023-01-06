---
layout: post
title: PyBullet-Gym for Drones
author: [00953103 蘇冠豪、00953106 張恩碩、01053203 邱顯淳]
category: [Lecture]
tags: [jekyll, ai]
---

### PyBullet-Gym for Drones (四軸無人機之強化學習)

**期末報告描述**

透過調整GUI按鍵數值(左右滑動)來控制四軸無人機飛行。

---

**開發過程:**
1. Cart-Pole System(車桿系統)<br>
2. 1D and 2D Quadrotor Systems(一維和二維四旋翼系統)<br>
3. Stabilization and Trajectory Tracking Tasks(穩定和軌跡跟踪任務)<br>
4. Safe-Control-Gym Extended API(安全控制運動擴展API)<br>
5. Computational Performance(計算性能)<br>

---

**執行環境:**
以下是使用版本供參考
* python 3.9.10
* gym-pybullet-drones-v0.5.2 (儲存.zip檔並解壓縮)
* PyCharm (選擇安裝程序中的所有選項並重新啟動)
* Visual Studio (選擇“使用 C++ 進行桌面開發”)

---

### PyCharm介紹

PyCharm: [PyCharm ](https://www.jetbrains.com/pycharm/)

**PyCharm** 是一個用於電腦編程的整合式開發環境，主要用於Python語言開發，由捷克公司JetBrains開發，提供代碼分析、圖形化除錯器，整合測試器、整合版本控制系統，並支援使用Django進行網頁開發。<br>
PyCharm是一個跨平台開發環境，擁有Microsoft Windows、macOS和Linux版本。社群版在Apache授權條款下釋出，另外還有專業版在專用授權條款下釋出，其擁有許多額外功能，比如Web開發、Python We框架、Python剖析器、遠端開發、支援資料庫與SQL等更多進階功能。

如果使用 Python 語言進行開發，PyCharm 支援下列幾種辨識功能
1. 項目和代碼導航：專門的項目視圖，文件結構視圖和文件、類、方法和用法之間的快速跳轉。<br>
2. Python 重構：包括重命名、提取方法、引入變量、引入常量、上拉、下壓等。<br>
3. 支持Web框架：Django，web2py和Flask。<br>
4. 集成的Python調試器。<br>
5. Google App Engine Python開發。<br>
6. 版本控制集成：Mercurial，Git，Subversion，Perforce和CVS的統一用戶界面，包含更改列表和合併。<br>
7. 它主要與許多其他面向Python的IDE競爭，包括Eclipse的PyDev和更廣泛的Komodo IDE。<br>

參考來源: [Python自習手札 ](https://ithelp.ithome.com.tw/articles/10196461)

---

### gym-pybullet-drones介紹

gym-pybullet-drones: [gym-pybullet-drones ](https://github.com/utiasDSL/gym-pybullet-drones)<br>
video: [Learning to Fly](https://www.youtube.com/watch?v=VdTsVu1HuYk&ab_channel=LearningSystemsandRoboticsLab)<br>
paper: [Safe Learning in Robotics](https://www.annualreviews.org/doi/abs/10.1146/annurev-control-042920-020211)<br>
paper: [Safe-Control-Gym](https://ieeexplore.ieee.org/abstract/document/9849119)<br>

PyBullet 是一個快速且易於使用的 Python 模塊，用於機器人仿真和機器學習，重點是模擬到真實的轉移。 使用 PyBullet，您可以從 URDF 加載關節體，SDF、MJCF 等文件格式。PyBullet 提供正向動力學模擬、逆向動力學計算、正向和逆向運動學、碰撞檢測和射線相交查詢。 Bullet Physics SDK 包括 PyBullet 機器人示例，例如模擬 Minitaur 四足動物、使用 TensorFlow 推理運行的人形機器人和 KUKA 手臂抓取物體。

gym-pybullet-drones 最近很多針對連續動作的 RL 研究都集中在策略梯度算法和 actor-critic 架構上。四旋翼飛行器是 (i) 一種易於理解的移動機器人平台，其 (ii) 控制可以構建為連續狀態和動作問題，但超出一維，(iii) 它增加了許多候選策略導致的複雜性不可恢復的狀態，違反了在蘊含的馬爾可夫鏈上存在靜止狀態分佈的假設。

|                                   | `gym-pybullet-drones` | [AirSim](https://github.com/microsoft/AirSim) | [Flightmare](https://github.com/uzh-rpg/flightmare) |
|---------------------------------: | :-------------------: | :-------------------------------------------: | :-------------------------------------------------: |
|                         *Physics* | PyBullet              | FastPhysicsEngine/PhysX                       | *Ad hoc*/Gazebo                                     |
|                       *Rendering* | PyBullet              | Unreal Engine 4                               | Unity                                               |
|                        *Language* | Python                | C++/C#                                        | C++/Python                                          |  
|           *RGB/Depth/Segm. views* | **Yes**               | **Yes**                                       | **Yes**                                             |
|             *Multi-agent control* | **Yes**               | **Yes**                                       | **Yes**                                             |
|                   *ROS interface* | ROS2/Python           | ROS/C++                                       | ROS/C++                                             |
|            *Hardware-In-The-Loop* | No                    | **Yes**                                       | No                                                  |
|         *Fully steppable physics* | **Yes**               | No                                            | **Yes**                                             |
|             *Aerodynamic effects* | Drag, downwash, ground| Drag                                          | Drag                                                |
|          *OpenAI [`Gym`](https://github.com/openai/gym/blob/master/gym/core.py) interface* | **Yes** | **[Yes](https://github.com/microsoft/AirSim/pull/3215)** | **Yes**                                             |
| *RLlib [`MultiAgentEnv`](https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py) interface* | **Yes** | No | No                           |

---

## 程式說明

程式的部分主要分成**辨識手掌**和**產生泡泡**還有**計分計時**三個部分，其中辨識手掌的部分主要參考自MediaPipe的官方文檔如下

---

### 官方文檔







## Student's Accomplishment
### [MediaPipe Hands poke bubbles](https://hahakevin45.github.io/AI/lecture/2022/12/08/Pose-Estimation.html)
<iframe width="854" height="480" src="https://www.youtube.com/embed/YJ_JCDBOgiE" title="MediaPipe Hands poke bubbles" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [使用 YOLOv5辨識水下魚種](https://alanlee0323.github.io/AI-course/lecture/2022/12/08/capstone-project.html)
<iframe width="723" height="482" src="https://www.youtube.com/embed/46wfrbQC8fI" title="影像偵測1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Teacher's samples

### Highway Traffice Analysis
**Kaggle:** [YOLOv5 Traffic Analysis](https://kaggle.com/rkuo2000/yolov5-traffic-analysis)<br>
use YOLOv5 to detect car/truck per frame, then analyze vehicle counts per lane and the estimated speed
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_traffic_analysis.jpg?raw=true)

---
### Facemask Detection
**Kaggle**: [YOLOv5 Facemask](https://www.kaggle.com/code/rkuo2000/yolov5-facemask)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_Facemask.jpg?raw=true)

---
### Face Identification
**Kaggle:** [FaceNet-PyTorch](https://kaggle.com/rkuo2000/FaceNet-PyTorch)<br>

---
### DeepFashion TryOn
**Kaggle:** [rkuo2000/deepfashion-tryon](https://www.kaggle.com/rkuo2000/deepfashion-tryon)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/VTON_ACGPN.jpg?raw=true)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*
