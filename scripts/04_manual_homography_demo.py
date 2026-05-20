#!/usr/bin/env python3

import os
import cv2
import numpy as np


IMAGE_PATH = os.path.expanduser(
    "~/mono_cone_perception/data/sample_images/left_sample_001.png"
)

OUTPUT_PATH = os.path.expanduser(
    "~/mono_cone_perception/outputs/manual_homography_demo.png"
)


# These are the 4 pixel points you clicked.
# Order:
# 1 = rear-left
# 2 = rear-right
# 3 = front-right
# 4 = front-left
image_points = np.array(
    [
        [301, 452],
        [952, 437],
        [766, 299],
        [520, 301],
    ],
    dtype=np.float32,
)


# Temporary fake world coordinates for learning.
# Coordinate convention:
# X = forward from car, meters
# Y = left from car, meters
#
# Later, these fake values will be replaced by real ArUco marker measurements.
world_points = np.array(
    [
        [1.0,  1.5],   # rear-left
        [1.0, -1.5],   # rear-right
        [6.0, -1.5],   # front-right
        [6.0,  1.5],   # front-left
    ],
    dtype=np.float32,
)


clicked_test_points = []


def image_to_world(pixel_point, H):
    """
    Convert image pixel (u, v) to ground-plane coordinate (X, Y).
    """
    pt = np.array([[pixel_point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    x_world, y_world = transformed[0, 0]
    return float(x_world), float(y_world)


def mouse_callback(event, x, y, flags, param):
    H = param

    if event == cv2.EVENT_LBUTTONDOWN:
        X, Y = image_to_world((x, y), H)
        clicked_test_points.append((x, y, X, Y))
        print(f"Clicked pixel: u={x}, v={y}  ->  X={X:.2f} m, Y={Y:.2f} m")


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    H_img_to_world, mask = cv2.findHomography(image_points, world_points)

    print("Image points:")
    print(image_points)

    print("\nWorld points:")
    print(world_points)

    print("\nHomography matrix H_img_to_world:")
    print(H_img_to_world)

    print("\nInstructions:")
    print("- Red points are the 4 calibration correspondences.")
    print("- Left-click any ground point to estimate X,Y.")
    print("- Press s to save debug image.")
    print("- Press q to quit.")

    cv2.namedWindow("Manual Homography Demo", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Homography Demo", mouse_callback, H_img_to_world)

    while True:
        display = image.copy()

        # Draw calibration points
        for i, (u, v) in enumerate(image_points.astype(int)):
            cv2.circle(display, (u, v), 7, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"P{i + 1}",
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # Draw clicked test points
        for u, v, X, Y in clicked_test_points:
            cv2.circle(display, (u, v), 5, (255, 0, 0), -1)
            cv2.putText(
                display,
                f"X={X:.1f}, Y={Y:.1f}",
                (u + 8, v + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Manual Homography Demo", display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            cv2.imwrite(OUTPUT_PATH, display)
            print(f"Saved debug image to: {OUTPUT_PATH}")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()