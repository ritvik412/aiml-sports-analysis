# aiml-sports-analysis
 # 🚀 AI/ML Football Analysis System

This repository implements an end-to-end AI/ML pipeline for detecting, tracking, and analyzing football players, referees, and the ball in match footage. Leveraging state-of-the-art object detection (YOLO), clustering (KMeans), optical flow, and perspective transformation, the system quantifies key performance metrics—such as ball acquisition percentage, player movement (in meters), speed, and distance covered—making it valuable for both beginners and experienced machine learning engineers.

# Key Features:-

1. Object Detection & Tracking: Detects players, referees, and footballs using YOL) (tracks them across frames).
2. Team Assignment: Segments and clusters jersey colors with KMeans to assign players to their respective teams.
3. Ball Acquisition Percentage: Computes the proportion of time each team possesses the ball.
4. Optical Flow for Camera Compensation: Measures camera movement between frames to isolate true player motion.
5. Perspective Transformation: Converts pixel coordinates into real‑world distances (meters) by modeling scene depth.
6. Speed & Distance Calculation: Derives player speed and total distance covered using transformed coordinates.

