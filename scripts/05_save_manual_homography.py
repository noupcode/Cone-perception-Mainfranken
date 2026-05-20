#!/usr/bin/env python3

import os
import cv2
import yaml
import numpy as np


OUTPUT_PATH = os.path.expanduser(
    "~/mono_cone_perception/config/homography_manual.yaml"
)


image_points = np.array(
    [
        [301, 452],
        [952, 437],
        [766, 299],
        [520, 301],
    ],
    dtype=np.float32,
)


world_points = np.array(
    [
        [1.0,  1.5],
        [1.0, -1.5],
        [6.0, -1.5],
        [6.0,  1.5],
    ],
    dtype=np.float32,
)


def main():
    H, mask = cv2.findHomography(image_points, world_points)

    data = {
        "description": "Temporary manual homography for learning only. Not final calibration.",
        "coordinate_frame": {
            "x": "forward from car",
            "y": "left from car",
            "unit": "meters",
        },
        "image_points": image_points.tolist(),
        "world_points": world_points.tolist(),
        "H_img_to_world": H.tolist(),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Saved manual homography to: {OUTPUT_PATH}")
    print("\nH_img_to_world:")
    print(H)


if __name__ == "__main__":
    main()