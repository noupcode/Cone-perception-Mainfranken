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
    "~/mono_cone_perception/outputs/fake_bbox_to_bev_debug.png"
)


# Same BEV settings as before
PIXELS_PER_METER = 100

X_MIN = 0.0
X_MAX = 8.0

Y_MIN = -3.0
Y_MAX = 3.0

BEV_WIDTH = int((Y_MAX - Y_MIN) * PIXELS_PER_METER)
BEV_HEIGHT = int((X_MAX - X_MIN) * PIXELS_PER_METER)


# Fake bounding boxes for visible cones in the sample image.
# Format:
# x_min, y_min, x_max, y_max, class_name
#
# These are approximate. You can adjust them later.
fake_boxes = [
    [260, 335, 360, 465, "orange"],   # left orange cone
    [335, 325, 420, 450, "blue"],     # left blue cone
    [875, 325, 965, 455, "yellow"],   # right yellow cone
    [930, 310, 1035, 455, "orange"],  # right orange cone
]


CLASS_DRAW_COLORS = {
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 140, 255),
    "big_orange": (0, 80, 255),
    "unknown": (255, 255, 255),
}


def load_homography(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    H = np.array(data["H_img_to_world"], dtype=np.float64)
    return H


def image_to_world(pixel_point, H):
    """
    Convert one image pixel point (u, v) to world ground point (X, Y).
    """
    pt = np.array([[pixel_point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    X, Y = transformed[0, 0]
    return float(X), float(Y)


def world_to_bev_pixel(X, Y):
    """
    Convert world coordinate (X forward, Y left) to BEV image pixel.
    """
    px = int((Y - Y_MIN) * PIXELS_PER_METER)
    py = int((X_MAX - X) * PIXELS_PER_METER)
    return px, py


def point_inside_bev(X, Y):
    return X_MIN <= X <= X_MAX and Y_MIN <= Y <= Y_MAX


def draw_bev_grid(bev):
    # Draw Y grid lines
    y = Y_MIN
    while y <= Y_MAX:
        px = int((y - Y_MIN) * PIXELS_PER_METER)
        cv2.line(bev, (px, 0), (px, BEV_HEIGHT), (70, 70, 70), 1)
        cv2.putText(
            bev,
            f"Y={y:.0f}",
            (px + 3, BEV_HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )
        y += 1.0

    # Draw X grid lines
    x = X_MIN
    while x <= X_MAX:
        py = int((X_MAX - x) * PIXELS_PER_METER)
        cv2.line(bev, (0, py), (BEV_WIDTH, py), (70, 70, 70), 1)
        cv2.putText(
            bev,
            f"X={x:.0f}",
            (5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )
        x += 1.0

    # Draw car origin at X=0, Y=0
    origin_px, origin_py = world_to_bev_pixel(0.0, 0.0)
    cv2.circle(bev, (origin_px, origin_py), 8, (255, 255, 255), -1)
    cv2.putText(
        bev,
        "car",
        (origin_px + 8, origin_py - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    H_img_to_world = load_homography(HOMOGRAPHY_PATH)

    camera_debug = image.copy()
    bev_debug = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    draw_bev_grid(bev_debug)

    print("Fake bbox → bottom-center pixel → world X,Y")
    print("------------------------------------------------")

    for box in fake_boxes:
        x_min, y_min, x_max, y_max, class_name = box

        color = CLASS_DRAW_COLORS.get(class_name, CLASS_DRAW_COLORS["unknown"])

        # Bottom-center pixel of bbox
        u = int((x_min + x_max) / 2)
        v = int(y_max)

        # Transform image point to world
        X, Y = image_to_world((u, v), H_img_to_world)

        print(
            f"{class_name:10s} bbox=({x_min},{y_min},{x_max},{y_max}) "
            f"bottom=({u},{v}) -> X={X:.2f} m, Y={Y:.2f} m"
        )

        # Draw bbox on camera image
        cv2.rectangle(camera_debug, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.circle(camera_debug, (u, v), 5, (0, 0, 255), -1)
        cv2.putText(
            camera_debug,
            f"{class_name}",
            (x_min, y_min - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.putText(
            camera_debug,
            f"X={X:.1f},Y={Y:.1f}",
            (x_min, y_max + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        # Draw point in BEV if inside display range
        if point_inside_bev(X, Y):
            px, py = world_to_bev_pixel(X, Y)

            cv2.circle(bev_debug, (px, py), 7, color, -1)
            cv2.putText(
                bev_debug,
                class_name,
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        else:
            print(f"  WARNING: {class_name} projected outside BEV display range.")

    # Resize camera image for side-by-side display
    camera_h = BEV_HEIGHT
    scale = camera_h / camera_debug.shape[0]
    camera_w = int(camera_debug.shape[1] * scale)
    camera_resized = cv2.resize(camera_debug, (camera_w, camera_h))

    combined = np.hstack([camera_resized, bev_debug])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, combined)

    print("------------------------------------------------")
    print(f"Saved debug image to: {OUTPUT_PATH}")

    cv2.namedWindow("Fake BBox to BEV Debug", cv2.WINDOW_NORMAL)
    cv2.imshow("Fake BBox to BEV Debug", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()