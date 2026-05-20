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
    "~/mono_cone_perception/outputs/real_bbox_to_bev_synced.png"
)


# BEV visualization settings
PIXELS_PER_METER = 100

X_MIN = 0.0
X_MAX = 8.0

Y_MIN = -3.0
Y_MAX = 3.0

BEV_WIDTH = int((Y_MAX - Y_MIN) * PIXELS_PER_METER)
BEV_HEIGHT = int((X_MAX - X_MIN) * PIXELS_PER_METER)


def load_homography_and_roi(path):
    """
    Load both:
    1. H_img_to_world: image pixel -> ground coordinate
    2. image_points: calibration polygon in the image

    The calibration polygon defines the image region where we trust the homography.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    H = np.array(data["H_img_to_world"], dtype=np.float64)
    image_points = np.array(data["image_points"], dtype=np.float32)

    return H, image_points


def point_inside_image_polygon(u, v, polygon_points):
    """
    Check whether image pixel (u, v) lies inside the calibration polygon.

    cv2.pointPolygonTest returns:
      positive -> inside
      zero     -> on edge
      negative -> outside
    """
    polygon = np.array(polygon_points, dtype=np.float32)
    result = cv2.pointPolygonTest(polygon, (float(u), float(v)), False)
    return result >= 0


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


def marker_to_bbox(marker):
    xs = [p.x for p in marker.points]
    ys = [p.y for p in marker.points]

    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def marker_array_stamp(msg, bag_time):
    """
    Foxglove ImageMarkerArray does not have a top-level header.
    Use the first marker header stamp when available.
    """
    if len(msg.markers) > 0:
        return msg.markers[0].header.stamp.to_sec()

    return bag_time.to_sec()


def draw_calibration_roi(image, roi_points):
    """
    Draw the calibration polygon on the camera image.
    """
    pts = roi_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    for i, (u, v) in enumerate(roi_points.astype(int)):
        cv2.circle(image, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"ROI{i+1}",
            (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )


def draw_bev_grid(bev):
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


def read_images_and_bboxes():
    """
    Read all image timestamps and bbox messages.

    For this bag duration, storing images in memory is acceptable.
    Later, for longer bags, we will stream instead of storing all images.
    """
    bridge = CvBridge()

    images = []
    bbox_messages = []

    with rosbag.Bag(BAG_PATH, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC, BBOX_TOPIC]):

            if topic == IMAGE_TOPIC:
                stamp = msg.header.stamp.to_sec()
                image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                images.append(
                    {
                        "stamp": stamp,
                        "image": image,
                    }
                )

            elif topic == BBOX_TOPIC:
                stamp = marker_array_stamp(msg, t)
                bbox_messages.append(
                    {
                        "stamp": stamp,
                        "msg": msg,
                        "num_markers": len(msg.markers),
                    }
                )

    return images, bbox_messages


def find_closest_image(images, target_stamp):
    best = None
    best_dt = None

    for item in images:
        dt = abs(item["stamp"] - target_stamp)

        if best is None or dt < best_dt:
            best = item
            best_dt = dt

    return best, best_dt


def choose_bbox_message(bbox_messages):
    """
    Choose a bbox message that has detections.

    For now, choose the first message with at least one marker.
    Later we can choose a specific timestamp or frame index.
    """
    for item in bbox_messages:
        if item["num_markers"] > 0:
            return item

    return None


def main():
    H_img_to_world, calibration_roi = load_homography_and_roi(HOMOGRAPHY_PATH)

    print("Reading images and bbox messages from bag...")
    images, bbox_messages = read_images_and_bboxes()

    print(f"Loaded {len(images)} images")
    print(f"Loaded {len(bbox_messages)} bbox messages")

    if len(images) == 0:
        raise RuntimeError("No images found.")

    if len(bbox_messages) == 0:
        raise RuntimeError("No bbox messages found.")

    bbox_item = choose_bbox_message(bbox_messages)

    if bbox_item is None:
        raise RuntimeError("No bbox messages with detections found.")

    bbox_stamp = bbox_item["stamp"]
    bbox_msg = bbox_item["msg"]

    image_item, dt = find_closest_image(images, bbox_stamp)

    image = image_item["image"]
    image_stamp = image_item["stamp"]

    print("\nSelected synchronized pair:")
    print(f"  bbox stamp:  {bbox_stamp:.6f}")
    print(f"  image stamp: {image_stamp:.6f}")
    print(f"  |dt|:        {dt:.6f} seconds")
    print(f"  markers:     {len(bbox_msg.markers)}")

    camera_debug = image.copy()
    draw_calibration_roi(camera_debug, calibration_roi)

    bev_debug = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    draw_bev_grid(bev_debug)

    print("\nSynced real bbox → ROI filter → bottom-center pixel → manual homography X,Y")
    print("--------------------------------------------------------------------------")

    valid_count = 0
    skipped_roi_count = 0
    skipped_bev_count = 0

    for i, marker in enumerate(bbox_msg.markers):
        if len(marker.points) < 4:
            continue

        x_min, y_min, x_max, y_max = marker_to_bbox(marker)

        u = int((x_min + x_max) / 2)
        v = int(y_max)

        if not point_inside_image_polygon(u, v, calibration_roi):
            print(
                f"{i:02d}: bbox=({x_min},{y_min},{x_max},{y_max}) "
                f"bottom=({u},{v}) -> skipped, outside calibration ROI"
            )
            skipped_roi_count += 1

            # Draw skipped boxes in gray
            cv2.rectangle(camera_debug, (x_min, y_min), (x_max, y_max), (120, 120, 120), 1)
            cv2.circle(camera_debug, (u, v), 4, (120, 120, 120), -1)
            continue

        X, Y = image_to_world((u, v), H_img_to_world)

        print(
            f"{i:02d}: bbox=({x_min},{y_min},{x_max},{y_max}) "
            f"bottom=({u},{v}) -> X={X:.2f} m, Y={Y:.2f} m"
        )

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
            valid_count += 1
        else:
            print("    WARNING: projected outside BEV range")
            skipped_bev_count += 1

    print("--------------------------------------------------------------------------")
    print(f"Valid BEV points drawn: {valid_count}")
    print(f"Skipped outside calibration ROI: {skipped_roi_count}")
    print(f"Skipped outside BEV display: {skipped_bev_count}")

    camera_h = BEV_HEIGHT
    scale = camera_h / camera_debug.shape[0]
    camera_w = int(camera_debug.shape[1] * scale)
    camera_resized = cv2.resize(camera_debug, (camera_w, camera_h))

    combined = np.hstack([camera_resized, bev_debug])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, combined)

    print(f"Saved output to: {OUTPUT_PATH}")

    cv2.namedWindow("Synced Real BBox to BEV with ROI Filter", cv2.WINDOW_NORMAL)
    cv2.imshow("Synced Real BBox to BEV with ROI Filter", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()