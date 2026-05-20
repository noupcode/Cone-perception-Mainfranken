#!/usr/bin/env python3

import os
import cv2
import yaml
import numpy as np


IMAGE_PATH = os.path.expanduser(
    "~/mono_cone_perception/data/sample_images/left_sample_001.png"
)

HOMOGRAPHY_PATH = os.path.expanduser(
    "~/mono_cone_perception/config/homography_manual.yaml"
)

OUTPUT_PATH = os.path.expanduser(
    "~/mono_cone_perception/outputs/birdseye_view_demo.png"
)


# Visual BEV settings
# We define a top-down canvas where:
# 100 pixels = 1 meter
PIXELS_PER_METER = 100

# Show X from 0 to 8 meters forward
X_MIN = 0.0
X_MAX = 8.0

# Show Y from -3 to +3 meters lateral
Y_MIN = -3.0
Y_MAX = 3.0

BEV_WIDTH = int((Y_MAX - Y_MIN) * PIXELS_PER_METER)
BEV_HEIGHT = int((X_MAX - X_MIN) * PIXELS_PER_METER)


def load_manual_points(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    image_points = np.array(data["image_points"], dtype=np.float32)
    world_points = np.array(data["world_points"], dtype=np.float32)

    return image_points, world_points


def world_to_bev_pixels(world_points):
    """
    Convert metric world coordinates (X forward, Y left)
    into BEV image pixel coordinates.

    BEV image convention:
    - image x-axis = lateral direction
    - image y-axis = forward direction, but flipped so farther forward is higher
    """
    bev_points = []

    for X, Y in world_points:
        px = (Y - Y_MIN) * PIXELS_PER_METER
        py = (X_MAX - X) * PIXELS_PER_METER
        bev_points.append([px, py])

    return np.array(bev_points, dtype=np.float32)


def draw_bev_grid(bev):
    """
    Draw a simple 1-meter grid on the BEV image.
    """
    # vertical grid lines for Y
    y = Y_MIN
    while y <= Y_MAX:
        px = int((y - Y_MIN) * PIXELS_PER_METER)
        cv2.line(bev, (px, 0), (px, BEV_HEIGHT), (80, 80, 80), 1)
        cv2.putText(
            bev,
            f"Y={y:.0f}",
            (px + 3, BEV_HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        y += 1.0

    # horizontal grid lines for X
    x = X_MIN
    while x <= X_MAX:
        py = int((X_MAX - x) * PIXELS_PER_METER)
        cv2.line(bev, (0, py), (BEV_WIDTH, py), (80, 80, 80), 1)
        cv2.putText(
            bev,
            f"X={x:.0f}",
            (5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        x += 1.0


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    image_points, world_points = load_manual_points(HOMOGRAPHY_PATH)

    # Convert world meter coordinates into BEV pixel coordinates
    bev_points = world_to_bev_pixels(world_points)

    # Homography for visualization:
    # image pixels -> BEV image pixels
    H_img_to_bev, _ = cv2.findHomography(image_points, bev_points)

    # Warp the original camera image into the BEV canvas
    bev = cv2.warpPerspective(image, H_img_to_bev, (BEV_WIDTH, BEV_HEIGHT))

    # Draw grid overlay
    draw_bev_grid(bev)

    # Draw calibration points in BEV
    for i, (px, py) in enumerate(bev_points.astype(int)):
        cv2.circle(bev, (px, py), 7, (0, 0, 255), -1)
        cv2.putText(
            bev,
            f"P{i+1}",
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, bev)

    print(f"Saved BEV image to: {OUTPUT_PATH}")
    print(f"BEV size: {BEV_WIDTH} x {BEV_HEIGHT}")
    print(f"Scale: {PIXELS_PER_METER} pixels = 1 meter")

    cv2.namedWindow("Bird's-Eye View Demo", cv2.WINDOW_NORMAL)
    cv2.imshow("Bird's-Eye View Demo", bev)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()