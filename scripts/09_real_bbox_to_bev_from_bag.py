#!/usr/bin/env python3

import os
import cv2
import yaml
import rosbag
import numpy as np
from cv_bridge import CvBridge


BAG_PATH = os.path.expanduser("~/mono_cone_data/bags/mono_cone_dev.bag")

IMAGE_TOPIC = "/zed2/zed_node/left/image_rect_color"
BBOX_TOPIC = "/stereo_cone_perception/bounding_boxes"

HOMOGRAPHY_PATH = os.path.expanduser(
    "~/mono_cone_perception/config/homography_manual.yaml"
)

OUTPUT_PATH = os.path.expanduser(
    "~/mono_cone_perception/outputs/real_bbox_to_bev_from_bag.png"
)


# BEV visualization settings
PIXELS_PER_METER = 100

X_MIN = 0.0
X_MAX = 8.0

Y_MIN = -3.0
Y_MAX = 3.0

BEV_WIDTH = int((Y_MAX - Y_MIN) * PIXELS_PER_METER)
BEV_HEIGHT = int((X_MAX - X_MIN) * PIXELS_PER_METER)


def load_homography(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return np.array(data["H_img_to_world"], dtype=np.float64)


def image_to_world(pixel_point, H):
    pt = np.array([[pixel_point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    X, Y = transformed[0, 0]
    return float(X), float(Y)


def world_to_bev_pixel(X, Y):
    px = int((Y - Y_MIN) * PIXELS_PER_METER)
    py = int((X_MAX - X) * PIXELS_PER_METER)
    return px, py


def point_inside_bev(X, Y):
    return X_MIN <= X <= X_MAX and Y_MIN <= Y <= Y_MAX


def draw_bev_grid(bev):
    # Y/lateral grid lines
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

    # X/forward grid lines
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

    # Car origin
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


def marker_to_bbox(marker):
    """
    Convert one Foxglove ImageMarker rectangle to bbox:
    x_min, y_min, x_max, y_max.

    The marker stores rectangle corners in marker.points.
    """
    xs = [p.x for p in marker.points]
    ys = [p.y for p in marker.points]

    x_min = int(min(xs))
    x_max = int(max(xs))
    y_min = int(min(ys))
    y_max = int(max(ys))

    return x_min, y_min, x_max, y_max


def read_first_image_and_bbox():
    """
    Read the first image and first bbox message from the bag.

    For this first demo, we just take the first available image and first
    available bbox message. Later we will synchronize them by timestamp.
    """
    bridge = CvBridge()

    image = None
    image_stamp = None

    bbox_msg = None
    bbox_stamp = None

    with rosbag.Bag(BAG_PATH, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC, BBOX_TOPIC]):

            if topic == IMAGE_TOPIC and image is None:
                image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                image_stamp = msg.header.stamp.to_sec()

            elif topic == BBOX_TOPIC and bbox_msg is None:
                bbox_msg = msg
                # ImageMarkerArray itself may not have a header,
                # so use first marker header if available.
                if len(msg.markers) > 0:
                    bbox_stamp = msg.markers[0].header.stamp.to_sec()
                else:
                    bbox_stamp = t.to_sec()

            if image is not None and bbox_msg is not None:
                break

    return image, image_stamp, bbox_msg, bbox_stamp


def main():
    H_img_to_world = load_homography(HOMOGRAPHY_PATH)

    image, image_stamp, bbox_msg, bbox_stamp = read_first_image_and_bbox()

    if image is None:
        raise RuntimeError("No image found in bag.")

    if bbox_msg is None:
        raise RuntimeError("No bounding box message found in bag.")

    print("Loaded from bag:")
    print(f"  image timestamp: {image_stamp}")
    print(f"  bbox timestamp:  {bbox_stamp}")
    print(f"  number of bbox markers: {len(bbox_msg.markers)}")

    camera_debug = image.copy()
    bev_debug = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    draw_bev_grid(bev_debug)

    print("\nReal bbox → bottom-center pixel → manual homography X,Y")
    print("------------------------------------------------------")

    for i, marker in enumerate(bbox_msg.markers):
        if len(marker.points) < 4:
            continue

        x_min, y_min, x_max, y_max = marker_to_bbox(marker)

        u = int((x_min + x_max) / 2)
        v = int(y_max)

        X, Y = image_to_world((u, v), H_img_to_world)

        print(
            f"{i:02d}: bbox=({x_min},{y_min},{x_max},{y_max}) "
            f"bottom=({u},{v}) -> X={X:.2f} m, Y={Y:.2f} m"
        )

        # Draw bbox
        color = (0, 255, 255)
        cv2.rectangle(camera_debug, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.circle(camera_debug, (u, v), 5, (0, 0, 255), -1)
        cv2.putText(
            camera_debug,
            f"{i}: X={X:.1f},Y={Y:.1f}",
            (x_min, max(20, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        # Draw in BEV
        if point_inside_bev(X, Y):
            px, py = world_to_bev_pixel(X, Y)
            cv2.circle(bev_debug, (px, py), 6, color, -1)
            cv2.putText(
                bev_debug,
                str(i),
                (px + 7, py - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        else:
            print(f"    WARNING: projected outside BEV range")

    # Combine camera image and BEV image
    camera_h = BEV_HEIGHT
    scale = camera_h / camera_debug.shape[0]
    camera_w = int(camera_debug.shape[1] * scale)
    camera_resized = cv2.resize(camera_debug, (camera_w, camera_h))

    combined = np.hstack([camera_resized, bev_debug])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, combined)

    print("------------------------------------------------------")
    print(f"Saved output to: {OUTPUT_PATH}")

    cv2.namedWindow("Real BBox to BEV From Bag", cv2.WINDOW_NORMAL)
    cv2.imshow("Real BBox to BEV From Bag", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()