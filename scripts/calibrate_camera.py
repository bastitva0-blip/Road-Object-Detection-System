#!/usr/bin/env python3
"""
Camera calibration utility for speed estimation.

Click two points in the frame that correspond to a known real-world distance
(e.g. lane width, or two marks on the road measured with a tape). The script
prints the resulting pixels_per_meter value to put into
config/default_config.yaml under speed_estimation.pixels_per_meter.

Usage:
    python scripts/calibrate_camera.py --image data/screenshots/frame_000001.png --distance 3.5
    python scripts/calibrate_camera.py --webcam 0 --distance 3.5
"""

import argparse
import sys

import cv2

points = []


def _on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")


def calibrate(frame, real_distance_meters: float) -> float:
    window = "Calibration - click 2 points, press 'q' when done"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_click)

    while True:
        display = frame.copy()
        for p in points:
            cv2.circle(display, p, 5, (0, 255, 0), -1)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) == 2:
            break

    cv2.destroyAllWindows()
    if len(points) != 2:
        print("Calibration cancelled: need exactly 2 points")
        sys.exit(1)

    (x1, y1), (x2, y2) = points
    pixel_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    pixels_per_meter = pixel_distance / real_distance_meters
    return pixels_per_meter


def main():
    parser = argparse.ArgumentParser(description="Calibrate pixels_per_meter for speed estimation")
    parser.add_argument("--image", type=str, help="Path to a reference image")
    parser.add_argument("--webcam", type=int, help="Webcam index to grab a reference frame from")
    parser.add_argument("--distance", type=float, required=True,
                        help="Real-world distance in meters between the two points you'll click")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}")
            sys.exit(1)
    elif args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Could not read from webcam {args.webcam}")
            sys.exit(1)
    else:
        print("Provide --image or --webcam")
        sys.exit(1)

    pixels_per_meter = calibrate(frame, args.distance)
    print(f"\npixels_per_meter = {pixels_per_meter:.2f}")
    print("Set this in config/default_config.yaml under speed_estimation.pixels_per_meter")


if __name__ == "__main__":
    main()
