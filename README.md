# aiml-sports-analysis
# AI/ML Football Analysis System ⚽🚀

<img src="Output_video.gif" width="1000" alt="Game plot animation"/>


This repository implements an end-to-end Computer Vision driven AI/ML pipeline for detecting, tracking, and analyzing football players, referees, and the ball in match footage. Leveraging state-of-the-art object detection (YOLO), clustering (KMeans), optical flow, and perspective transformation, the system quantifies key performance metrics—such as ball acquisition percentage, player movement (in meters), speed, and distance covered—making it valuable for both beginners and experienced machine learning engineers.

## 📖 Project Overview:

1. Object Detection & Tracking: Detects players, referees, and footballs using YOLO (tracks them across frames).
2. Team Assignment: Segments and clusters jersey colors with KMeans to assign players to their respective teams.
3. Ball Acquisition Percentage: Computes the proportion of time each team possesses the ball.
4. Optical Flow for Camera Compensation: Measures camera movement between frames to isolate true player motion.
5. Perspective Transformation: Converts pixel coordinates into real‑world distances (meters) by modeling scene depth.
6. Speed & Distance Calculation: Derives player speed and total distance covered using transformed coordinates.


## 🛠️ Tech Stack:

| Layer           | Technologies & Tools                                                                                                     |
|-----------------|----------------------------------------------------------                                                                |
| **Backend**     | Python 3.11.3                                                                                                            |
|                 | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)(Use YOLOv8 for models)                                    |
|                 | [OpenCV](https://opencv.org/)                                                                                            |
|                 | [Supervision](https://github.com/roboflow/supervision)                                                                   |
|                 | RoboFlow API                                                                                          |                                         
| **Modules**     | • `trackers/` (ByteTrack integration)                                                                                    |
|                 | • `team_assigner/`, `player_ball_assigner/`                                                                              |
|                 | • `camera_movement_estimator/`, `view_transformer/`                                                                      |
|                 | • `speed_and_distance_estimator/`                                                                                        |
| **Data & I/O**  | [Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)(Use YOLOv5)|
|                 | NumPy, Pandas                                                                                                            |
|**Visualization**| Matplotlib, scikit‑learn (K‑Means)                                                                                       |
| **Notebook**    | Jupyter Notebook                                                                                                         |
| **Dev Tools**   | Git, Visual Studio Code                                                                                                  |


## 🚀 Setup Guide:

1. **Clone the repo**  
   ```bash
   git clone https://github.com/ritvik412/aiml-sports-analysis.git
   cd aiml-sports-analysis

2. **Create and Activate a virtual Environment**
   ```bash
     python3 -m venv venv
     source venv/bin/activate   # macOS/Linux
     venv\Scripts\activate      # Windows

3. **Install Dependencies & Get your RoboFlow API key**
4.   **Sample Input Video**
     [input video](08fd33_4.mp4)

## 📝 Use Cases:
1. **Performance Analytics & Coaching Insights**
   
   •Player Workload Tracking: Automatically compute distance covered and speed profiles for each player to spot fatigue patterns or injury risk.

   •Tactical Breakdown: Visualize ball‑possession heatmaps and passing lanes in a top‑down view, helping coaches adapt formations and strategies mid‑match.

2. **Automated Broadcast Augmentation**
   
   •Live Graphics Overlays: Integrate your YOLO‑based detection and tracking into live streams to display real‑time IDs, jersey colors, and speed metrics on screen.

   •Instant Replays with Metrics: Auto‑generate video clips tagged with possession percentages or sprint speeds for in‑broadcast replays.

## 📄 License
This project is licensed under the MIT License. See LICENSE for details.
