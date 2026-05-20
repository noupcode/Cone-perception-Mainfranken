#!/usr/bin/env python3

import sys
import cv2
import numpy as np

def main():
    print("Python:", sys.version)
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    print("Has ArUco:", hasattr(cv2, "aruco"))

    if hasattr(cv2, "aruco"):
        print("ArUco module OK")
    else:
        print("ArUco module missing")

if __name__ == "__main__":
    main()
