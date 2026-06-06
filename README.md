# Monocular Cone Localization for Formula Student Driverless

This project explores a lightweight monocular perception pipeline for estimating cone positions using a single camera image.

The motivation is simple: in Formula Student Driverless, the car needs reliable cone positions in the vehicle frame. Traditional pipelines often use stereo depth, LiDAR, or full 3D perception. This project investigates how far we can go using only a monocular camera image, 2D cone bounding boxes, and a calibrated image-to-ground homography.

The current implementation runs both offline on a real ROS bag and live as a ROS node.

---

## What the project does

The pipeline takes cone bounding boxes in the camera image and estimates their ground-plane positions:

```text
ZED left camera image
+ cone bounding boxes
+ image-to-ground homography
→ cone X/Y positions on the ground
```
For each cone detection, the bottom-center of the bounding box is used as the approximate ground contact point of the cone. This pixel is then projected into metric ground coordinates using a homography.

The output is a 2D cone position estimate in a vehicle/camera ground frame:
```text
X = forward
Y = left
Z = 0
```
# Current status

The project currently supports:

Reading real Formula Student Driverless ROS bag data
Extracting ZED left camera images and camera info
Parsing cone bounding boxes from /stereo_cone_perception/bounding_boxes
Projecting cone bottom-center pixels into ground coordinates
Exporting monocular cone estimates to CSV
Exporting stereo cone reference positions to CSV
Comparing monocular estimates against stereo reference cones
Running a live ROS node on rosbag playback
Publishing monocular cone positions as a ROS PointCloud
Publishing a debug image with bounding boxes, bottom-center points, ROI, and projected X/Y labels

The current system uses a manually initialized homography as a baseline. A proper ArUco-marker-based calibration is the next accuracy improvement step.

Example pipeline
```text
rosbag play
    ↓
/zed2/zed_node/left/image_rect_color
/stereo_cone_perception/bounding_boxes
    ↓
17_mono_cone_ros_node.py
    ↓
/mono_cone_perception/cones
/mono_cone_perception/debug_image
Dataset / ROS topics
```
The development ROS bag contains real vehicle data, including:
```
/zed2/zed_node/left/image_rect_color
/zed2/zed_node/left/camera_info
/stereo_cone_perception/bounding_boxes
/stereo_cone_perception/cones
/lidar/cone_position_cloud
```
The current monocular pipeline uses:

Input image:
```
  /zed2/zed_node/left/image_rect_color
```
Input 2D cone boxes:
```
  /stereo_cone_perception/bounding_boxes
```
Reference for validation:
```
  /stereo_cone_perception/cones
```
Important note: the current localization part is monocular, but the 2D bounding boxes are still taken from the existing stereo cone perception output. A future step is replacing this with an independent monocular cone detector.

# Repository structure
```
mono_cone_perception/
├── config/
│   ├── camera.json
│   ├── homography_manual.yaml
│   ├── homography_team.yaml
│   └── aruco_markers.yaml
│
├── data/
│   ├── sample_images/
│   └── calibration_images/
│
├── docs/
│   ├── project_plan.md
│   ├── data_inventory.md
│   ├── geometry_notes.md
│   └── session_log.md
│
├── outputs/
│   ├── aruco_markers/
│   ├── test_manual/
│   └── test_manual_no_roi/
│
└── scripts/
    ├── 01_extract_sample_from_bag.py
    ├── 05_save_manual_homography.py
    ├── 06_birdseye_view_demo.py
    ├── 10_real_bbox_to_bev_synced.py
    ├── 11_export_projected_cones_csv.py
    ├── 12_export_stereo_cones_csv.py
    ├── 13_compare_mono_vs_stereo.py
    ├── 14_generate_aruco_markers.py
    ├── 15_detect_aruco_in_image.py
    ├── 16_compute_homography_from_aruco.py
    └── 17_mono_cone_ros_node.py
```

# Requirements

Tested with:

  Ubuntu 20.04
  ROS Noetic
  Python 3.8
  OpenCV
  NumPy
  PyYAML
  rosbag
  cv_bridge
  foxglove_msgs


  # Limitations

This project is still under development.

 Current limitations:

  The current homography is manually initialized and not final
  Accuracy is limited by calibration quality
  2D bounding boxes currently come from an existing perception topic
  The system is not yet a full standalone monocular detector
  Matching against stereo reference is currently nearest-neighbor based
  No temporal tracking is implemented yet
  No final vehicle integration or Jetson optimization yet
  Next steps

# Planned next milestones:

  Compute a real ArUco-based homography from measured ground markers
  Re-run the accuracy comparison with homography_aruco.yaml
  Add a live bird's-eye-view debug visualization
  Improve validation with per-frame matching and error plots
  Convert the scripts into a proper ROS package
  Replace existing bounding boxes with an independent monocular cone detector
  Add cone tracking using Kalman filtering and Hungarian assignment
  Test real-time performance on the target vehicle computer
