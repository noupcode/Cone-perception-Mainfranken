#!/usr/bin/env python3

import os
import csv
import rosbag


BAG_PATH = os.path.expanduser("~/mono_cone_data/bags/mono_cone_dev.bag")

STEREO_CONES_TOPIC = "/stereo_cone_perception/cones"

OUTPUT_CSV = os.path.expanduser(
    "~/mono_cone_perception/outputs/stereo_cones_reference.csv"
)


def get_channel_values(msg, channel_name):
    """
    sensor_msgs/PointCloud stores extra data in channels.
    Each channel has:
      name
      values

    In our bag, useful channels are:
      color
      confidence
    """
    for channel in msg.channels:
        if channel.name == channel_name:
            return list(channel.values)

    return None


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    total_messages = 0
    total_cones = 0

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "stamp",
                "frame_id",
                "cone_id",
                "X_stereo",
                "Y_stereo",
                "Z_stereo",
                "color",
                "confidence",
            ]
        )

        print(f"Reading bag: {BAG_PATH}")
        print(f"Reading topic: {STEREO_CONES_TOPIC}")

        with rosbag.Bag(BAG_PATH, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[STEREO_CONES_TOPIC]):
                total_messages += 1

                stamp = msg.header.stamp.to_sec()
                frame_id = msg.header.frame_id

                color_values = get_channel_values(msg, "color")
                confidence_values = get_channel_values(msg, "confidence")

                for cone_id, point in enumerate(msg.points):
                    color = ""
                    confidence = ""

                    if color_values is not None and cone_id < len(color_values):
                        color = color_values[cone_id]

                    if confidence_values is not None and cone_id < len(confidence_values):
                        confidence = confidence_values[cone_id]

                    writer.writerow(
                        [
                            stamp,
                            frame_id,
                            cone_id,
                            point.x,
                            point.y,
                            point.z,
                            color,
                            confidence,
                        ]
                    )

                    total_cones += 1

    print("\nExport complete.")
    print("--------------------------------")
    print(f"Total stereo cone messages: {total_messages}")
    print(f"Total stereo cones:         {total_cones}")
    print(f"Saved CSV to:               {OUTPUT_CSV}")


if __name__ == "__main__":
    main()