#!/usr/bin/env python3

import os
import cv2


IMAGE_PATH = os.path.expanduser(
    "~/mono_cone_perception/data/sample_images/left_sample_001.png"
)

points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: u={x}, v={y}")


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    print("Instructions:")
    print("- Click 4 points on the ground.")
    print("- Prefer visible lane-line/ground reference points.")
    print("- Press r to reset.")
    print("- Press q to quit and print final points.")

    cv2.namedWindow("Click Ground Points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click Ground Points", mouse_callback)

    while True:
        display = image.copy()

        for i, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(
                display,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Click Ground Points", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
            print("Reset points.")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    print("\nFinal clicked points:")
    for i, (x, y) in enumerate(points):
        print(f"{i + 1}: [{x}, {y}]")


if __name__ == "__main__":
    main()