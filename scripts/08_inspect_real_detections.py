#!/usr/bin/env python3

import os
import rosbag

BAG_PATH = os.path.expanduser("~/mono_cone_data/bags/mono_cone_dev.bag")

TOPICS = [
    "/stereo_cone_perception/bounding_boxes",
    "/stereo_cone_perception/cones",
]


def print_first_msg(topic_name):
    print("\n" + "=" * 80)
    print(f"Topic: {topic_name}")
    print("=" * 80)

    with rosbag.Bag(BAG_PATH, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            print(msg)
            return

    print("No messages found.")


def main():
    for topic in TOPICS:
        print_first_msg(topic)


if __name__ == "__main__":
    main()