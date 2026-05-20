#!/usr/bin/env python3

import os
import cv2


IMAGE_PATH = os.path.expanduser(
    "~/mono_cone_perception/data/sample_images/left_sample_001.png"
)


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    print(f"Loaded image: {IMAGE_PATH}")
    print(f"Image shape: {image.shape}")

    cv2.namedWindow("Left Sample Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Left Sample Image", image)

    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()