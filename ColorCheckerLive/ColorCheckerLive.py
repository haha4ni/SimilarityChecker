#!/usr/bin/env python3

import os
os.putenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import sys
import cv2
import json
import time
import logging
import argparse
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

##########################
# Argument and Config    #
##########################
g_color_tolerance = {
    "h_tolerance": 15,      # Hue tolerance (0-180), default 15
    "s_tolerance": 60,      # Saturation tolerance (0-255), default 60
    "v_tolerance": 60       # Value tolerance (0-255), default 60
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Similarity Check Tool - Live")
    parser.add_argument("-sr", "--set-roi", action="store_true", help="Capture frame from camera and set ROI")
    parser.add_argument("-ur", "--use-roi", action="store_true", help="Compare live video using saved ROI")
    parser.add_argument("-ds", "--device-sn", type=str, default="null", help="Device serial number (default: null)")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("-tt", "--test-time", type=int, default=3, help="Test time in seconds (default: 3)")
    parser.add_argument("-fp", "--find-pattern", action="store_true", help="Find reference ROI in camera frame automatically")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

def setup_logging(device_serial):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"app_debug_log_{device_serial}_{timestamp}.txt")
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().addHandler(file_handler)

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error("Invalid config.json. Starting with empty config.")
    return {}

def save_config(config, config_path='config.json'):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

##########################
# Score Functions      #
##########################
def calculate_color_similarity(img1, img2):
    """
    Calculate color similarity using HSV color space
    Args:
        img1, img2: Input images (BGR format)
        h_tolerance: Hue tolerance (0-180)
        s_tolerance: Saturation tolerance (0-255)
        v_tolerance: Value tolerance (0-255)
    Returns:
        similarity_score: Float between 0-1, higher means more similar
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    h_tolerance = g_color_tolerance["h_tolerance"]
    s_tolerance = g_color_tolerance["s_tolerance"]
    v_tolerance = g_color_tolerance["v_tolerance"]

    # Convert to HSV
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Calculate mean HSV values
    mean_hsv1 = cv2.mean(hsv1)[:3]  # Get only H, S, V (ignore alpha)
    mean_hsv2 = cv2.mean(hsv2)[:3]

    # Calculate differences
    h_diff = abs(mean_hsv1[0] - mean_hsv2[0])
    # Handle hue wraparound (0 and 180 are close in hue space)
    h_diff = min(h_diff, 180 - h_diff)

    s_diff = abs(mean_hsv1[1] - mean_hsv2[1])
    v_diff = abs(mean_hsv1[2] - mean_hsv2[2])

    # Calculate similarity scores for each channel (1.0 = perfect match, 0.0 = maximum difference)
    h_similarity = max(0, 1 - (h_diff / h_tolerance)) if h_tolerance > 0 else 1
    s_similarity = max(0, 1 - (s_diff / s_tolerance)) if s_tolerance > 0 else 1
    v_similarity = max(0, 1 - (v_diff / v_tolerance)) if v_tolerance > 0 else 1

    # Combine similarities (weighted average - hue is most important for color)
    combined_similarity = (h_similarity * 0.5 + s_similarity * 0.3 + v_similarity * 0.2)

    return combined_similarity

##########################
# Main Logic             #
##########################
def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    args = parse_arguments()
    setup_logging(args.device_sn)

    logging.debug("*** ColorCheckerLive v0.0.4 ***")
    logging.debug(f"Args: {args}")

    config = load_config()
    logging.debug(f"Config: {json.dumps(config, indent=2)}")

    # Get color tolerance settings from config
    g_color_tolerance = config.get("color_tolerance", {
        "h_tolerance": 15,      # Hue tolerance (0-180), default 15
        "s_tolerance": 60,      # Saturation tolerance (0-255), default 60
        "v_tolerance": 60       # Value tolerance (0-255), default 60
    })

    # Color similarity threshold (0.0-1.0, higher means stricter)
    color_similarity_threshold = config.get("pass_criteria", {}).get("color_similarity_threshold", 0.7)

    # Initialize camera manager
    from utils.CameraManager import CameraManager
    camera_manager = CameraManager(logging, camera_index=args.camera, frame_width=1920, frame_height=1080)

    try:
        if args.set_roi:
            logging.info("=== SET ROI MODE ===")

            from utils.RoiHelper import RoiHelper
            roi_helper = RoiHelper(logging, camera_manager)

            # ROI selection
            rois = roi_helper.set_roi()
            if rois:
                config["roi"] = rois
                save_config(config)
                logging.info("ROI saved to config.json")
            else:
                logging.warning("No ROI selected")

        elif args.use_roi:
            logging.info("=== USE ROI MODE ===")

            device_serial = args.device_sn.strip()
            if not device_serial:
                logging.error("Please provide device serial number")
                return

            roi_data = config.get("roi", [])
            if not roi_data:
                logging.error("No ROI found in config, please run --set-roi first")
                return

            logging.info(f"Starting color detection, test time: {args.test_time} seconds")
            logging.info(f"Color tolerance - H: {g_color_tolerance['h_tolerance']}, S: {g_color_tolerance['s_tolerance']}, V: {g_color_tolerance['v_tolerance']}")
            logging.info(f"Color similarity threshold: {color_similarity_threshold}")

            from utils.RoiHelper import RoiHelper
            roi_helper = RoiHelper(logging, camera_manager)

            result, snapshot = roi_helper.use_roi(device_serial, args.test_time, roi_data, calculate_color_similarity, color_similarity_threshold, args.find_pattern, args.debug)

            logging.info(f"Device SN: {device_serial} | Test result: {result}")

            # Save result image
            save_dir = os.path.join(os.getcwd(), "result_img_capture")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{device_serial}_{result}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), snapshot)
            logging.debug(f"Result image saved: {filename}")

            cv2.destroyAllWindows()
        else:
            logging.error("Please specify --set-roi or --use-roi parameter")

    except Exception as e:
        logging.error(f"Program execution error: {e}", exc_info=True)

    finally:
        # Clean up resources
        camera_manager.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
