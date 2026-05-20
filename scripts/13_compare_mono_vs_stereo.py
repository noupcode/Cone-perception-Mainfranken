#!/usr/bin/env python3

import os
import csv
import math
from collections import defaultdict


MONO_CSV = os.path.expanduser(
    "~/mono_cone_perception/outputs/projected_cones_manual_homography.csv"
)

STEREO_CSV = os.path.expanduser(
    "~/mono_cone_perception/outputs/stereo_cones_reference.csv"
)

OUTPUT_CSV = os.path.expanduser(
    "~/mono_cone_perception/outputs/mono_vs_stereo_comparison.csv"
)


# Matching settings
# We match mono detections to stereo cones from the closest timestamp.
MAX_TIME_DIFF = 0.05      # seconds
MAX_MATCH_DIST = 2.0      # meters, loose for now because manual homography is rough


def read_mono_csv(path):
    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "stamp": float(row["bbox_stamp"]),
                    "detection_id": int(row["detection_id"]),
                    "x_min": int(row["x_min"]),
                    "y_min": int(row["y_min"]),
                    "x_max": int(row["x_max"]),
                    "y_max": int(row["y_max"]),
                    "u": int(row["u_bottom_center"]),
                    "v": int(row["v_bottom_center"]),
                    "X_mono": float(row["X_mono"]),
                    "Y_mono": float(row["Y_mono"]),
                }
            )

    return rows


def read_stereo_csv(path):
    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "stamp": float(row["stamp"]),
                    "frame_id": row["frame_id"],
                    "cone_id": int(row["cone_id"]),
                    "X_stereo": float(row["X_stereo"]),
                    "Y_stereo": float(row["Y_stereo"]),
                    "Z_stereo": float(row["Z_stereo"]),
                    "color": row["color"],
                    "confidence": row["confidence"],
                }
            )

    return rows


def group_by_stamp(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["stamp"]].append(row)

    return dict(grouped)


def find_closest_stamp(target_stamp, available_stamps):
    best_stamp = None
    best_dt = None

    for stamp in available_stamps:
        dt = abs(stamp - target_stamp)

        if best_stamp is None or dt < best_dt:
            best_stamp = stamp
            best_dt = dt

    return best_stamp, best_dt


def distance_2d(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def range_distance(x, y):
    return math.sqrt(x ** 2 + y ** 2)


def find_nearest_stereo_cone(mono_row, stereo_candidates, used_stereo_ids):
    """
    Find nearest stereo cone in X,Y space for one mono detection.
    Avoid reusing the same stereo cone within the same timestamp.
    """
    best = None
    best_dist = None

    for stereo in stereo_candidates:
        stereo_key = (stereo["stamp"], stereo["cone_id"])

        if stereo_key in used_stereo_ids:
            continue

        dist = distance_2d(
            mono_row["X_mono"],
            mono_row["Y_mono"],
            stereo["X_stereo"],
            stereo["Y_stereo"],
        )

        if best is None or dist < best_dist:
            best = stereo
            best_dist = dist

    return best, best_dist


def compute_metrics(matches):
    if len(matches) == 0:
        return None

    errors_2d = [m["error_2d"] for m in matches]
    errors_x = [abs(m["error_x"]) for m in matches]
    errors_y = [abs(m["error_y"]) for m in matches]
    errors_range = [abs(m["error_range"]) for m in matches]

    mean_2d = sum(errors_2d) / len(errors_2d)
    mean_x = sum(errors_x) / len(errors_x)
    mean_y = sum(errors_y) / len(errors_y)
    mean_range = sum(errors_range) / len(errors_range)

    rmse_2d = math.sqrt(sum(e ** 2 for e in errors_2d) / len(errors_2d))

    sorted_errors = sorted(errors_2d)
    median_2d = sorted_errors[len(sorted_errors) // 2]
    p95_2d = sorted_errors[int(0.95 * (len(sorted_errors) - 1))]

    return {
        "count": len(matches),
        "mean_2d": mean_2d,
        "median_2d": median_2d,
        "p95_2d": p95_2d,
        "rmse_2d": rmse_2d,
        "mean_abs_x": mean_x,
        "mean_abs_y": mean_y,
        "mean_abs_range": mean_range,
    }


def main():
    print("Loading CSV files...")
    mono_rows = read_mono_csv(MONO_CSV)
    stereo_rows = read_stereo_csv(STEREO_CSV)

    print(f"Mono projected detections: {len(mono_rows)}")
    print(f"Stereo reference cones:    {len(stereo_rows)}")

    stereo_by_stamp = group_by_stamp(stereo_rows)
    stereo_stamps = sorted(stereo_by_stamp.keys())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    matches = []
    skipped_time = 0
    skipped_distance = 0

    # To avoid reusing stereo cones, track usage per stereo timestamp.
    used_stereo_ids = set()

    for mono in mono_rows:
        closest_stereo_stamp, dt = find_closest_stamp(mono["stamp"], stereo_stamps)

        if closest_stereo_stamp is None or dt > MAX_TIME_DIFF:
            skipped_time += 1
            continue

        stereo_candidates = stereo_by_stamp[closest_stereo_stamp]

        nearest_stereo, match_dist = find_nearest_stereo_cone(
            mono, stereo_candidates, used_stereo_ids
        )

        if nearest_stereo is None or match_dist > MAX_MATCH_DIST:
            skipped_distance += 1
            continue

        used_stereo_ids.add(
            (nearest_stereo["stamp"], nearest_stereo["cone_id"])
        )

        error_x = mono["X_mono"] - nearest_stereo["X_stereo"]
        error_y = mono["Y_mono"] - nearest_stereo["Y_stereo"]
        error_2d = distance_2d(
            mono["X_mono"],
            mono["Y_mono"],
            nearest_stereo["X_stereo"],
            nearest_stereo["Y_stereo"],
        )

        mono_range = range_distance(mono["X_mono"], mono["Y_mono"])
        stereo_range = range_distance(
            nearest_stereo["X_stereo"],
            nearest_stereo["Y_stereo"],
        )
        error_range = mono_range - stereo_range

        matches.append(
            {
                "mono_stamp": mono["stamp"],
                "stereo_stamp": nearest_stereo["stamp"],
                "dt": dt,
                "mono_detection_id": mono["detection_id"],
                "stereo_cone_id": nearest_stereo["cone_id"],
                "u": mono["u"],
                "v": mono["v"],
                "X_mono": mono["X_mono"],
                "Y_mono": mono["Y_mono"],
                "X_stereo": nearest_stereo["X_stereo"],
                "Y_stereo": nearest_stereo["Y_stereo"],
                "Z_stereo": nearest_stereo["Z_stereo"],
                "stereo_color": nearest_stereo["color"],
                "stereo_confidence": nearest_stereo["confidence"],
                "error_x": error_x,
                "error_y": error_y,
                "error_2d": error_2d,
                "mono_range": mono_range,
                "stereo_range": stereo_range,
                "error_range": error_range,
            }
        )

    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = [
            "mono_stamp",
            "stereo_stamp",
            "dt",
            "mono_detection_id",
            "stereo_cone_id",
            "u",
            "v",
            "X_mono",
            "Y_mono",
            "X_stereo",
            "Y_stereo",
            "Z_stereo",
            "stereo_color",
            "stereo_confidence",
            "error_x",
            "error_y",
            "error_2d",
            "mono_range",
            "stereo_range",
            "error_range",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for match in matches:
            writer.writerow(match)

    metrics = compute_metrics(matches)

    print("\nComparison complete.")
    print("--------------------------------")
    print(f"Mono detections:              {len(mono_rows)}")
    print(f"Stereo cones:                 {len(stereo_rows)}")
    print(f"Matched pairs:                {len(matches)}")
    print(f"Skipped due to time:          {skipped_time}")
    print(f"Skipped due to distance:      {skipped_distance}")
    print(f"Output CSV:                   {OUTPUT_CSV}")

    if metrics is None:
        print("\nNo matches found. Try increasing MAX_TIME_DIFF or MAX_MATCH_DIST.")
        return

    print("\nError metrics:")
    print("--------------------------------")
    print(f"Mean 2D error:                {metrics['mean_2d']:.3f} m")
    print(f"Median 2D error:              {metrics['median_2d']:.3f} m")
    print(f"95th percentile 2D error:     {metrics['p95_2d']:.3f} m")
    print(f"RMSE 2D error:                {metrics['rmse_2d']:.3f} m")
    print(f"Mean absolute X error:        {metrics['mean_abs_x']:.3f} m")
    print(f"Mean absolute Y error:        {metrics['mean_abs_y']:.3f} m")
    print(f"Mean absolute range error:    {metrics['mean_abs_range']:.3f} m")

    print("\nImportant note:")
    print("These metrics use the temporary manual homography, not final ArUco calibration.")
    print("So treat this as a pipeline sanity check, not final system accuracy.")


if __name__ == "__main__":
    main()