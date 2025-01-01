# Human Fall Detection Using CNN, MediaPipe, and YOLOv5

This repository implements a fall detection system that uses a combination of Convolutional Neural Networks (CNN), MediaPipe for pose estimation, and YOLOv5 for detecting multiple people in video frames. The system processes video input, extracts human pose features, and classifies each frame as either "Fall" or "Non-Fall" based on the detected poses.

## Overview

Falls are a significant health risk, especially for elderly individuals. This project aims to build a real-time human fall detection system using modern computer vision techniques. It combines:

- **CNN (Convolutional Neural Networks)**: Used for classifying images into "Fall" or "Non-Fall" categories.
- **MediaPipe**: A framework by Google that provides real-time pose estimation. We use it to extract human pose landmarks from video frames.
- **YOLOv5 (You Only Look Once)**: A deep learning model used for detecting and localizing multiple people in the video frames.

The system works in real-time, detecting and classifying falls as people are detected in the frame.

## Project Structure

This repository consists of three main scripts that handle different aspects of the fall detection system:

1. **Fall Detection Model (CNN + MediaPipe)**:
   - This script trains a Convolutional Neural Network (CNN) to classify images as either "Fall" or "Non-Fall" using pose landmarks extracted by MediaPipe.
   - The model is trained using labeled images from a dataset of fall and non-fall events.

2. **Real-Time Fall Detection (Pose + YOLOv5)**:
   - This script enables real-time fall detection using a webcam. It integrates YOLOv5 for detecting people and MediaPipe for extracting pose landmarks.
   - For each detected person, the model classifies them as either "Fall" or "Non-Fall" based on their pose.

3. **Multiple Person Detection and Fall Classification**:
   - This script builds on the previous two by detecting and classifying multiple people in real-time.
   - It processes the video frame by frame, detects multiple individuals, extracts their pose landmarks using MediaPipe, and classifies their fall status.

---

## Code Explanation

### 1. **Fall Detection Model (CNN + MediaPipe)**

The fall detection model is built using a CNN that learns to classify fall and non-fall events based on pose landmarks. Here’s how the process works:

- **Pose Estimation**: MediaPipe is used to extract pose landmarks from each image. These landmarks are key points that describe the position of different body parts (such as the shoulders, elbows, knees, etc.).
- **Data Preparation**: The dataset consists of images of both falling and non-falling individuals. Each image is labeled as "Fall" or "Non-Fall".
- **Training the CNN**: The CNN model is trained on these pose features extracted from the images. The goal is to learn patterns in the body posture that signify a fall event.
- **Model Output**: Once trained, the model can predict whether an input image depicts a fall or a non-fall.

The trained CNN model is saved as `Model.h5`, which is later used for real-time predictions in the other scripts.

### 2. **Real-Time Fall Detection (Pose + YOLOv5)**

In this script, we perform real-time fall detection using a webcam feed. The steps are:

- **Webcam Feed**: The video feed is captured from the webcam using OpenCV.
- **Person Detection (YOLOv5)**: YOLOv5 is used to detect people in each frame. YOLOv5 is a state-of-the-art object detection model that localizes multiple objects (in this case, people) in a given frame.
- **Pose Estimation (MediaPipe)**: For each person detected by YOLOv5, MediaPipe is used to extract their pose landmarks.
- **Fall Classification (CNN)**: The pose landmarks are fed into the pre-trained CNN model (`Model.h5`), which classifies whether the person is in a "Fall" or "Non-Fall" state.
- **Real-Time Output**: The classification result is displayed in real-time on the video feed, with the bounding box around the person and a label ("Fall" or "Non-Fall").

This allows the system to process live video and detect falls as they occur.

### 3. **Multiple Person Detection and Fall Classification**

This script is an extension of the previous one, but it can handle multiple people in a single frame. Here's how it works:

- **Multiple Person Detection (YOLOv5)**: Instead of detecting just one person, YOLOv5 detects all the people in the frame.
- **Pose Estimation (MediaPipe)**: For each person detected, MediaPipe extracts the pose landmarks. This allows us to assess the pose of each person individually.
- **Fall Classification (CNN)**: Each person’s pose landmarks are passed into the trained CNN model to determine if they are falling or not.
- **Multiple Output**: The system will draw a bounding box and display a "Fall" or "Non-Fall" label for each detected person.

This functionality is essential for scenarios where multiple people are present in the scene, ensuring that each person’s fall status is identified.

---

## Installation

Before running the scripts, you need to install the required dependencies. You can install them using `pip`:

```bash
pip install opencv-python mediapipe tensorflow torch yolov5
