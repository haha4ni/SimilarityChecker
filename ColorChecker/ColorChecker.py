#!/usr/bin/env python3

import os
import cv2
import sys
import json
import pytz
import argparse
import logging
import numpy as np
from datetime import datetime


##############################
# Config & Argument Parsing #
##############################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Comparison Tool with ROI Support")
    parser.add_argument("ref_image", help="Path to the reference image")
    parser.add_argument("test_image", nargs="?", help="Folder containing test images")
    parser.add_argument("--use-roi", action="store_true", help="Only compare within saved ROI")
    parser.add_argument("--set-roi", action="store_true", help="Force reselecting ROI and updating config")
    return parser.parse_args()


def load_config(config_path='config.json'):
    default_config = {
        "roi": None
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error("Invalid config.json. Using defaults.")
    return default_config


def save_config(config, config_path='config.json'):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


###################
# ROI Operations  #
###################
def select_roi(image):
    clone = image.copy()
    roi = cv2.selectROI("Select ROI", clone, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow("Select ROI")
    if roi[2] == 0 or roi[3] == 0:
        return None
    return roi


def crop_and_resize(image, roi, size=(1280, 720)):
    x, y, w, h = roi
    cropped = image[y:y+h, x:x+w]
    return cv2.resize(cropped, size)


def resize_image(image, size=(1280, 720)):
    return cv2.resize(image, size)


def detect_led_color(img):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Get average HSV values
    avg_hue = np.mean(hsv[:,:,0])
    avg_sat = np.mean(hsv[:,:,1])
    avg_val = np.mean(hsv[:,:,2])

    print("Avg H is:", avg_hue)
    print("Avg S is:", avg_sat)
    print("Avg V is:", avg_val)

    # Determine color based on average HSV values
    if avg_val > 200 and avg_sat < 50:
        return "White"
    elif 0 <= avg_hue <= 10 or avg_hue >= 160:
        return "Red"
    elif 35 <= avg_hue <= 85:
        return "Green"
    elif 90 <= avg_hue <= 130:
        return "Blue"
    else:
        return "Unknown"


#############
# Main App  #
#############
def main():
    args = parse_arguments()
    if not args.set_roi and not args.test_image:
        print("Error: test_image is required unless --set-roi is specified.")
        sys.exit(1)
    config = load_config()

    timezone = pytz.timezone("Asia/Taipei")
    timestamp = datetime.now(timezone).strftime("%Y%m%d_%H%M%S %Z")
    log_folder = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_folder, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_folder, f'check_monitor_{timestamp}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    report_logger = logging.getLogger('report')
    report_logger.addHandler(logging.FileHandler(os.path.join(log_folder, f'test_report_{timestamp}.log')))
    report_logger.setLevel(logging.INFO)
    report_logger.propagate = False

    logging.info("*** ColorChecker v0.0.0 Beta ***")
    logging.info(f"Args: ref_image={args.ref_image}, test_image={args.test_image}, use_roi={args.use_roi}, set_roi={args.set_roi}")
    logging.info(f"Config: {json.dumps(config, indent=2)}")

    ref_img = cv2.imread(args.ref_image)
    if ref_img is None:
        logging.error("Failed to load reference image.")
        sys.exit(1)

    ref_img = resize_image(ref_img)

    roi = None
    if args.set_roi:
        logging.info("Please select ROI on reference image.")
        roi = select_roi(ref_img)
        if roi is None:
            logging.error("No ROI selected. Exiting.")
            sys.exit(1)
        config['roi'] = {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]}
        save_config(config)
        logging.info("ROI saved to config.json.")
        exit(0)

    if args.use_roi:
        roi_data = config.get("roi")
        if not roi_data:
            logging.error("ROI not set. Use --set-roi first.")
            sys.exit(1)
        roi = (roi_data['x'], roi_data['y'], roi_data['w'], roi_data['h'])
        logging.info(f"Using ROI from config: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    for file in os.listdir(args.test_image):
        test_path = os.path.join(args.test_image, file)

        test_img = cv2.imread(test_path)
        if test_img is None:
            logging.warning(f"Failed to load image: {file}")
            continue

        test_img = resize_image(test_img)
        if args.use_roi:
            test_img = crop_and_resize(test_img, roi)

        color = detect_led_color(test_img)
        print("Color is:", color)

    logging.info("====================================================================")


if __name__ == "__main__":
    main()
