# Monocular Cone Perception Project Plan

## Goal

Estimate Formula Student Driverless cone positions in metric ground-plane coordinates using a single rectified monocular camera, homography calibration, cone detection, and multi-object tracking.

The system should run in real time at >= 30 Hz on the team PC.

## Motivation

The current car uses ZED2 stereo and/or LiDAR-dependent perception. The goal of this project is to reduce bus load and remove runtime LiDAR dependency by replacing metric cone localization with a monocular geometric method.

## Core Idea

A single monocular camera cannot recover arbitrary 3D depth from one image.

However, if the point of interest lies on a known plane, such as the ground plane, then the image pixel can be mapped to a real-world ground coordinate using a homography.

For cones, the bottom-center of the bounding box is used as an approximation of the cone's ground contact point.

## High-Level Pipeline

1. Receive rectified left camera image.
2. Detect cones using YOLO.
3. For each bounding box, compute the bottom-center pixel.
4. Use an image-to-ground homography to transform that pixel to metric X,Y.
5. Track cone positions over time using Kalman filters and data association.
6. Publish cone positions in ROS.
7. Visualize camera detections and bird's-eye-view cone map.

## Project Phases

### Phase 0: Data and Frame Inventory

Understand available rosbags, topics, camera calibration, existing cone outputs, and coordinate frames.

### Phase 1: Manual Homography Demo

Use one image and manually selected correspondences to map image pixels to ground-plane coordinates.

### Phase 2: BEV Debug Visualization

Use fake bounding boxes to draw camera detections and corresponding metric cone points in a local BEV plot.

### Phase 3: ArUco Marker Calibration

Detect ArUco markers, use their known physical corner coordinates, compute homography automatically, and save calibration.

### Phase 4: Offline Cone Localization

Use recorded images and existing bounding boxes or YOLO detections to produce monocular cone positions offline.

### Phase 5: Validation

Compare monocular homography cone positions against stereo pipeline outputs, LiDAR outputs, or manual measurements.

### Phase 6: Tracking

Implement Kalman filter prediction, Hungarian assignment, track creation, deletion, and smoothing.

### Phase 7: ROS Node

Integrate the pipeline into a ROS Noetic node with image input, detector input, cone output, and debug visualization topics.

### Phase 8: Real-Time Optimization

Profile and optimize on Jetson AGX Orin to reach >= 30 Hz.

### Phase 9: Portfolio Polish

Create clear documentation, diagrams, demo videos, validation plots, and interview explanations.

## Output Coordinate Frame

TODO:
- Frame name:
- Origin:
- +X direction:
- +Y direction:
- +Z direction:

## Important Open Questions

1. Are ArUco markers used only for calibration or continuously during runtime?
2. Are markers placed on the ground or mounted on the car?
3. What is the expected accuracy requirement in meters?
4. What is the maximum useful cone localization range?
5. What message type should the final ROS node publish?
6. Which frame should the output cone positions use?