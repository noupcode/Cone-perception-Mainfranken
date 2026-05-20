#!/usr/bin/env python3

import os
import csv
import cv2
import yaml
import rosbag
import numpy as np


BAG_PATH = os.path.expanduser("~/mono_cone_data/bags/mono_cone_dev.bag")

BBOX_TOPIC = "/stereo_cone_perception/bounding_boxes"

HOMOGRAPHY_PATH = os.path.expanduser(
    "~/mono_cone_perception/config/homography_manual.yaml"
)

OUTPUT_CSV = os.path.expanduser(
    "~/mono_cone_perception/outputs/projected_cones_manual_homography.csv"
)


def load_homography_and_roi(path):
    """
    Load:
    - H_img_to_world: maps image pixel (u, v) to ground coordinate (X, Y)
    - image_points: manual calibration polygon in the image
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    H = np.array(data["H_img_to_world"], dtype=np.float64)
    image_points = np.array(data["image_points"], dtype=np.float32)

    return H, image_points


def point_inside_image_polygon(u, v, polygon_points):
    """
    Check whether image pixel (u, v) lies inside the calibration polygon.
    """
    polygon = np.array(polygon_points, dtype=np.float32)
    result = cv2.pointPolygonTest(polygon, (float(u), float(v)), False)
    return result >= 0


def image_to_world(pixel_point, H):
    """
    Convert image pixel (u, v) to ground-plane coordinate (X, Y).
    """
    pt = np.array([[pixel_point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    X, Y = transformed[0, 0]
    return float(X), float(Y)


def marker_array_stamp(msg, bag_time):
    """
    Foxglove ImageMarkerArray does not have a top-level header.
    So we use the first marker's header stamp when available.
    """
    if len(msg.markers) > 0:
        return msg.markers[0].header.stamp.to_sec()

    return bag_time.to_sec()


def marker_to_bbox(marker):
    """
    Convert one Foxglove ImageMarker rectangle to bbox:
    x_min, y_min, x_max, y_max.
    """
    xs = [p.x for p in marker.points]
    ys = [p.y for p in marker.points]

    x_min = int(min(xs))
    y_min = int(min(ys))
    x_max = int(max(xs))
    y_max = int(max(ys))

    return x_min, y_min, x_max, y_max


def main():
    H_img_to_world, calibration_roi = load_homography_and_roi(HOMOGRAPHY_PATH)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    total_bbox_messages = 0
    total_markers = 0
    exported_count = 0
    skipped_roi_count = 0
    skipped_invalid_count = 0

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "bbox_stamp",
                "detection_id",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "u_bottom_center",
                "v_bottom_center",
                "X_mono",
                "Y_mono",
                "inside_roi",
            ]
        )

        print(f"Reading bag: {BAG_PATH}")
        print(f"Reading topic: {BBOX_TOPIC}")

        with rosbag.Bag(BAG_PATH, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[BBOX_TOPIC]):
                total_bbox_messages += 1

                bbox_stamp = marker_array_stamp(msg, t)

                for detection_id, marker in enumerate(msg.markers):
                    total_markers += 1

                    if len(marker.points) < 4:
                        skipped_invalid_count += 1
                        continue

                    x_min, y_min, x_max, y_max = marker_to_bbox(marker)

                    u = int((x_min + x_max) / 2)
                    v = int(y_max)

                    inside_roi = point_inside_image_polygon(
                        u, v, calibration_roi
                    )

                    if not inside_roi:
                        skipped_roi_count += 1
                        continue

                    X, Y = image_to_world((u, v), H_img_to_world)

                    writer.writerow(
                        [
                            bbox_stamp,
                            detection_id,
                            x_min,
                            y_min,
                            x_max,
                            y_max,
                            u,
                            v,
                            X,
                            Y,
                            int(inside_roi),
                        ]
                    )

                    exported_count += 1

    print("\nExport complete.")
    print("--------------------------------")
    print(f"Total bbox messages:         {total_bbox_messages}")
    print(f"Total markers seen:          {total_markers}")
    print(f"Exported projected cones:    {exported_count}")
    print(f"Skipped outside ROI:         {skipped_roi_count}")
    print(f"Skipped invalid markers:     {skipped_invalid_count}")
    print(f"Saved CSV to:                {OUTPUT_CSV}")


if __name__ == "__main__":
    main()