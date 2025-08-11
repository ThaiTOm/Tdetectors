# 🚦🚗 **Tdetectors - Smart City Systems** 🚓🔍 — Detection, Recognition & Identification Vehicle for 1

This project is a part of the **Tdetectors** initiative 🏙️💡, which aims to enhance smart city systems through advanced detection and tracking technologies 🤖📹.

**Contributors:** 👨‍💻💪

* Nguyễn Duy Thái (N22DCCN077) 
* Trần Nguyễn Sơn Thành (N22DCCN078) 
* Cao Duy Thái (N22DCCN076) 

---

## 🎥 **Demo**

[![Watch the video](demo/images/Screenshot%202025-07-29%20100652.png)](https://www.youtube.com/watch?v=AktD6WMdBYs)
[![Watch the video](demo/images/Screenshot%202025-07-29%20101050.png)](https://www.youtube.com/watch?v=Z6NlvcCQByA)
![Tracking for Multi-camera Multi Object](demo/images/Screenshot%202025-07-29%20120028.png)

---

### 📋 **Project Overview: Multi-Camera Vehicle Tracking with OpenVINO™** 🚘📡

This project implements a robust **Multi-Camera Multi-Object Tracking (MC-MOT)** system 🛰️ designed to detect, track, and re-identify vehicles 🚙🚓 across a wide network of traffic cameras 🎯, focusing on Ho Chi Minh City 🏙️.

🎯 **Main Goal:** Solve the **vehicle re-identification (Re-ID)** challenge — assign a consistent, unique ID 🆔 to each vehicle and track it across multiple camera views.

⚡ **Key Achievement:** Optimized the entire AI pipeline with **Intel's OpenVINO™ Toolkit** 🛠️💨, boosting performance from a sluggish model 🐌 (0.17 FPS) to a blazing-fast system 🚀 (15.23 FPS) on a CPU.

---

### 🛠️ **Technologies Used**

* 🖥️ **Intel® OpenVINO™ Toolkit** — Core optimization 🔧, speeding inference by \~**90x** 🚄.
* 🚗 **Vehicle & License Plate Detection (YOLOv12)** — Finds vehicles & plates in each frame 🎯.
* 🔤 **License Plate Recognition (CCT)** — Reads plate characters 📖👀.
* 🎨 **Vehicle Re-Identification (Multi-Branch Learning)** — Creates unique feature embeddings 🧩 for matching vehicles.
* 🎯 **Tracking Algorithm (BoT-SORT)** — Keeps stable tracking even with occlusions 🕵️‍♂️.

---

## ⚙️ **Installation**

```bash
pip install -r requirements.txt
```

---

## ▶️ **Usage**

```bash
# 🚘 Vehicle & License Plate Detection:
python output/Vehicle_and_License.py --source="exp.mp4"
# 📁 Result saved in the `output` folder.
```

```bash
# 🖼️ Cluster images from 2 hours of Ho Chi Minh traffic video:
python models/ReID/cluster_from_camera_hcm.py
```

---

## 📊 **OpenVINO Results (CPU Only)**

| 📏 Metric      | 🐢 PyTorch CPU | ⚡ OpenVINO CPU | 📈 Improvement |
| -------------- | -------------- | -------------- | -------------- |
| **Throughput** | **0.17 FPS**   | **15.23 FPS**  | **\~8858% 🚀** |
| **Latency**    | **5720.48 ms** | **327.33 ms**  | **↓ \~94.3%**  |



If you want, I can also **add playful emoji tags** inside your code snippets and headings to make even the technical sections fun but still readable. Would you like me to do that?
