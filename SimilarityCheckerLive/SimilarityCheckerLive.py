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
def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Similarity Check Tool - Live")
    parser.add_argument("-sr", "--set-roi", action="store_true", help="Capture frame from camera and set ROI")
    parser.add_argument("-ur", "--use-roi", action="store_true", help="Compare live video using saved ROI")
    parser.add_argument("-ds", "--device-sn", type=str, default="null", help="Device serial number (default: null)")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("-tt", "--test-time", type=int, default=3, help="Test time in seconds (default: 3)")
    parser.add_argument("-fp", "--find-pattern", action="store_true", help="Find reference ROI in camera frame automatically")
    parser.add_argument("-sm", "--score-method", type=str, default="SSIM", help="Scoring method. Options: SSIM, HIST")
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
def calculate_image_similarity_by_ssim(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        similarity = ssim(gray1, gray2)
        return similarity
    else:
        return ssim(img1, img2)

def calculate_image_similarity_by_histogram(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist1, None, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, None, 0, 1, cv2.NORM_MINMAX)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return (1 - similarity)

##########################
# Main Logic             #
##########################
def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    args = parse_arguments()
    setup_logging(args.device_sn)

    logging.debug("*** SimilarityCheckerLive v0.0.5 ***")
    logging.debug(f"Args: {args}")

    config = load_config()
    logging.debug(f"Config: {json.dumps(config, indent=2)}")

    similarity_threshold = config.get("pass_criteria", {}).get("similarity_threshold_histogram", 0.54)

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

            from utils.RoiHelper import RoiHelper
            roi_helper = RoiHelper(logging, camera_manager)

            if args.score_method == "SSIM":
                score_func = calculate_image_similarity_by_ssim
            elif args.score_method == "HIST":
                score_func = calculate_image_similarity_by_histogram

            result, snapshot = roi_helper.use_roi(device_serial, args.test_time, roi_data, score_func, similarity_threshold, args.find_pattern, args.debug)

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
