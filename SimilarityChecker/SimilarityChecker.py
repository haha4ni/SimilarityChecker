#!/usr/bin/env python3

import os
import glob
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
    parser.add_argument("-ri", "--ref-image", help="Path to the reference image")
    parser.add_argument("-ti", "--test-image", help="Path to the folder containing test images or a single test image")
    parser.add_argument("-sr", "--set-roi", action="store_true", help="Selecting ROI and updating config")
    parser.add_argument("-ur", "--use-roi", action="store_true", help="Only compare within saved ROI")
    parser.add_argument("-cr", "--check-roi", action="store_true", help="Show the saved ROI with reference image")
    return parser.parse_args()


def load_config(config_path='config.json'):
    default_config = {
        "pass_criteria": {"similarity_threshold_histogram": 0.54},
        "debug": {
            "dump_image_when_fail": True,
            "force_dump_image": False
        },
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
def select_single_roi(image):
    clone = image.copy()
    roi = cv2.selectROI("Select ROI", clone, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow("Select ROI")
    if roi[2] == 0 or roi[3] == 0:
        return None
    return roi


def select_multiple_rois(image):
    clone = image.copy()
    rois_with_labels = []

    print("[INFO] Hold left key to select ROI, Press Enter or Space to set ROI. Press ESC to exit.")

    while True:
        roi = cv2.selectROI("Select ROI", clone, fromCenter=False, showCrosshair=False)
        if roi[2] == 0 or roi[3] == 0:
            break

        label = input(f"Please enter the label for this ROI (x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}): ").strip()
        if not label:
            label = f"ROI_{len(rois_with_labels)}"

        rois_with_labels.append({"rect": roi, "label": label})

        x, y, w, h = roi
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(clone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.destroyWindow("Select ROI")
    return rois_with_labels if rois_with_labels else None


def crop_and_resize(image, roi, size=(1280, 720)):
    x, y, w, h = roi
    cropped = image[y:y+h, x:x+w]
    return cv2.resize(cropped, size)


def resize_image(image, size=(1280, 720)):
    return cv2.resize(image, size)


######################
# Histogram Analysis #
######################
def generate_bgr_histogram_image(image_src):
    bgr_planes = cv2.split(image_src)
    histSize = 256
    histRange = (0, 256)
    histImage = np.zeros((720, 1280, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for i, plane in enumerate(bgr_planes):
        hist = cv2.calcHist([plane], [0], None, [histSize], histRange)
        cv2.normalize(hist, hist, 0, 720, cv2.NORM_MINMAX)
        for j in range(1, histSize):
            cv2.line(histImage,
                     (5*(j-1), 720 - int(hist[j-1].item())),
                     (5*j, 720 - int(hist[j].item())),
                     colors[i], 2)
    return histImage


def calculate_image_similarity_by_histogram(img1, img2):
    img1_split = cv2.split(img1)
    img2_split = cv2.split(img2)
    similarity = 0
    for ch1, ch2 in zip(img1_split, img2_split):
        hist1 = cv2.calcHist([ch1], [0], None, [256], [0, 255])
        hist2 = cv2.calcHist([ch2], [0], None, [256], [0, 255])
        sim = sum(1 - abs(h1 - h2) / max(h1, h2) if h1 != h2 else 1 for h1, h2 in zip(hist1, hist2))
        similarity += sim / len(hist1)
    return float(similarity / 3)


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

    logging.info("*** SimilarityChecker v0.0.1 ***")
    logging.info(f"Args: ref_image={args.ref_image}, test_image={args.test_image}, use_roi={args.use_roi}, set_roi={args.set_roi}")
    logging.info(f"Config: {json.dumps(config, indent=2)}")

    similarity_threshold = config['pass_criteria']['similarity_threshold_histogram']

    ref_img = cv2.imread(args.ref_image)
    if ref_img is None:
        logging.error("Failed to load reference image. Please check the path.")
        sys.exit(1)

    ref_img = resize_image(ref_img)
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)

    rois = None
    if args.set_roi:
        logging.info("Please select ROI on reference image.")
        rois = select_multiple_rois(ref_img)
        if rois is None:
            logging.error("No ROI selected. Exiting.")
            sys.exit(1)

        config['roi'] = rois
        save_config(config)
        logging.info("ROI saved to config.json.")
        exit(0)

    if args.use_roi:
        rois = config.get("roi")
        if not rois:
            logging.error("ROI not set. Use --set-roi first.")
            sys.exit(1)

        # [(rect, label)]
        rois = [(tuple(roi["rect"]), roi.get("label", f"ROI_{i}")) for i, roi in enumerate(rois)]
        if rois:
            if args.check_roi:
                logging.info("Show ROIs on reference image.")
                for i, (roi, label) in enumerate(rois):
                    x, y, w, h = roi
                    roi_img = ref_img[y:y+h, x:x+w]
                    cv2.imshow(f"{label}.jpg", roi_img)
                cv2.waitKey(0)
                exit(0)

            ref_img = [ref_img[y:y+h, x:x+w] for ((x, y, w, h), label) in rois]
            ref_hsv = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in ref_img]

    test_image_paths = []
    if os.path.isdir(args.test_image):
        test_image_paths = [os.path.join(args.test_image, f)
                            for f in os.listdir(args.test_image)
                            if os.path.isfile(os.path.join(args.test_image, f))]
    elif os.path.isfile(args.test_image):
        test_image_paths = [args.test_image]
    else:
        logging.error(f"Invalid test image path: {args.test_image}")
        sys.exit(1)

    for test_path in test_image_paths:
        if os.path.abspath(test_path) == os.path.abspath(args.ref_image):
            continue

        test_img = cv2.imread(test_path)
        if test_img is None:
            logging.warning(f"Failed to load image: {test_path}")
            continue

        test_img = resize_image(test_img)

        if args.use_roi:
            test_img_rois = [test_img[y:y+h, x:x+w] for ((x, y, w, h), label) in rois]
            test_hsv_rois = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in test_img_rois]
            test_gray_rois = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in test_img_rois]

            sift = cv2.SIFT_create()

            failed_roi_infos = []

            for i, ((roi, label), ref_roi_hsv, test_roi_hsv, test_roi_gray) in enumerate(zip(rois, ref_hsv, test_hsv_rois, test_gray_rois)):
                score = calculate_image_similarity_by_histogram(ref_roi_hsv, test_roi_hsv)
                score = float(score[0]) if isinstance(score, (np.ndarray, list)) else float(score)
                keypoints = sift.detect(test_roi_gray, None)

                if score < similarity_threshold or len(keypoints) == 0:
                    failed_roi_infos.append({
                        "index": i,
                        "label": label,
                        "score": score,
                        "keypoints": len(keypoints)
                    })

            result = "Pass" if not failed_roi_infos else "Fail"
        else:
            test_hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
            test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            score = calculate_image_similarity_by_histogram(ref_hsv, test_hsv)
            score = float(score[0]) if isinstance(score, (np.ndarray, list)) else float(score)
            sift = cv2.SIFT_create()
            keypoints = sift.detect(test_gray, None)

            result = "Pass" if score >= similarity_threshold and len(keypoints) > 0 else "Fail"

        filename = os.path.basename(test_path)
        logging.info(f"{filename}: {result}")
        report_logger.info(f"Test image: {filename}      Test result: {result}")

        if result == "Fail":
            if args.use_roi:
                for info in failed_roi_infos:
                    logging.warning(f"ROI {info['index']} ({info['label']}) failed: histogram={info['score']:.3f}, keypoints={info['keypoints']}")
                    report_logger.warning(f"        Failed ROIs:")
                    report_logger.warning(f"            Index: {info['index']}, Label: {info['label']}\n")
            else:
                logging.debug(f"HSV histogram: {score:.3f}, SIFT keypoints size: {len(keypoints)}")

            if config['debug']['dump_image_when_fail'] or config['debug']['force_dump_image']:
                fail_dir = os.path.join(os.getcwd(), "fail_img_analysis")
                os.makedirs(fail_dir, exist_ok=True)

                if args.use_roi:
                    for i, ((roi, label), roi_img) in enumerate(zip(rois, test_img_rois)):
                        is_fail = any(f["index"] == i for f in failed_roi_infos)
                        tag = "FAIL" if is_fail else "PASS"
                        hist_img = generate_bgr_histogram_image(roi_img)
                        cv2.imwrite(os.path.join(fail_dir, f"{filename}_roi_{i}_{label}_{tag}.jpg"), hist_img)
                else:
                    cv2.imwrite(os.path.join(fail_dir, f"{filename}_hist.jpg"), generate_bgr_histogram_image(test_img))

    logging.info("====================================================================")


if __name__ == "__main__":
    main()
