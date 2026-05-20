#!/usr/bin/env python3

import os
import json
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge

BAG_PATH = os.path.expanduser("~/mono_cone_data/bags/mono_cone_dev.bag")
IMAGE_TOPIC = "/zed2/zed_node/left/image_rect_color"
CAMERA_INFO_TOPIC = "/zed2/zed_node/left/camera_info"

OUT_IMAGE = os.path.expanduser("~/mono_cone_perception/data/sample_images/left_sample_001.png")
OUT_CAMERA_JSON = os.path.expanduser("~/mono_cone_perception/config/camera.json")


def camera_info_to_dict(msg):
    return {
        "header": {
            "seq": msg.header.seq,
            "stamp": {
                "secs": msg.header.stamp.secs,
                "nsecs": msg.header.stamp.nsecs,
            },
            "frame_id": msg.header.frame_id,
        },
        "width": msg.width,
        "height": msg.height,
        "distortion_model": msg.distortion_model,
        "D": list(msg.D),
        "K": list(msg.K),
        "R": list(msg.R),
        "P": list(msg.P),
        "fx": msg.K[0],
        "fy": msg.K[4],
        "cx": msg.K[2],
        "cy": msg.K[5],
    }


def main():
    os.makedirs(os.path.dirname(OUT_IMAGE), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_CAMERA_JSON), exist_ok=True)

    bridge = CvBridge()

    saved_image = False
    saved_camera = False

    print(f"Reading bag: {BAG_PATH}")

    with rosbag.Bag(BAG_PATH, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC, CAMERA_INFO_TOPIC]):

            if topic == CAMERA_INFO_TOPIC and not saved_camera:
                camera_data = camera_info_to_dict(msg)

                with open(OUT_CAMERA_JSON, "w") as f:
                    json.dump(camera_data, f, indent=2)

                print(f"Saved camera info to: {OUT_CAMERA_JSON}")
                print(f"Camera frame: {camera_data['header']['frame_id']}")
                print(f"Image size: {camera_data['width']} x {camera_data['height']}")
                print(f"fx={camera_data['fx']}, fy={camera_data['fy']}, cx={camera_data['cx']}, cy={camera_data['cy']}")

                saved_camera = True

            if topic == IMAGE_TOPIC and not saved_image:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv2.imwrite(OUT_IMAGE, cv_img)

                print(f"Saved sample image to: {OUT_IMAGE}")
                print(f"Image shape: {cv_img.shape}")

                saved_image = True

            if saved_image and saved_camera:
                break

    if not saved_image:
        print("ERROR: No image message found.")

    if not saved_camera:
        print("ERROR: No camera_info message found.")


if __name__ == "__main__":
    main()
