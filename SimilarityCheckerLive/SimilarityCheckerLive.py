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
    parser.add_argument("--item-verify", type=int, help="Verify specific item by number")
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
        elif args.item_verify is not None:
            logging.info(f"=== ITEM VERIFY MODE: item {args.item_verify} ===")
            import subprocess
            # 讀取 JSON 檔
            json_path = os.path.join(os.getcwd(), "benchmark", "Test items.json")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load item json: {e}")
                return

            # 找到對應編號
            item = next((x for x in items if x.get("No") == args.item_verify), None)
            if not item:
                logging.error(f"Item No {args.item_verify} not found in json.")
                return

            commands = item.get("Commands", [])
            if not commands:
                logging.warning(f"No commands for item No {args.item_verify}.")
                return

            for cmd_str in commands:
                full_cmd = f'adb shell "{cmd_str}"'
                logging.info(f"Running: {full_cmd}")
                try:
                    result = subprocess.run(full_cmd, capture_output=True, text=True, shell=True)
                    logging.info(f"ADB output: {result.stdout.strip()}")
                    if result.stderr:
                        logging.error(f"ADB error: {result.stderr.strip()}")
                except Exception as e:
                    logging.error(f"Failed to run adb command: {e}")

            # 顯示 sample 圖片 & ROI 比對
            sample_img_path = os.path.join(os.getcwd(), "benchmark", "sample", f"{args.item_verify}.jpg")
            roi_rects = item.get("RoiRects", [])
            if os.path.exists(sample_img_path):
                logging.info(f"Show sample image: {sample_img_path}")
                img = cv2.imread(sample_img_path)
                if img is not None:
                    # 預覽縮放比例，目標高度 480p
                    preview_height = 480
                    preview_scale = preview_height / img.shape[0]
                    preview_size = (int(img.shape[1] * preview_scale), preview_height)
                    img_preview = cv2.resize(img, preview_size, interpolation=cv2.INTER_AREA)
                    # 畫出所有 ROI 框
                    for rect in roi_rects:
                        x = int(rect["x"] * preview_scale)
                        y = int(rect["y"] * preview_scale)
                        w = int(rect["w"] * preview_scale)
                        h = int(rect["h"] * preview_scale)
                        cv2.rectangle(img_preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    logging.info(f"Preview image scale: {preview_scale:.3f}, resized to: {preview_size}")
                    cv2.imshow(f"Sample {args.item_verify}", img_preview)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f"Sample {args.item_verify}")
                else:
                    logging.warning(f"Failed to load image: {sample_img_path}")

            # ROI 比對功能
            if os.path.exists(sample_img_path) and roi_rects:
                from utils.RoiHelper import RoiHelper
                roi_helper = RoiHelper(logging, camera_manager)
                # 準備 roi_data 格式
                roi_data = []
                for i, rect in enumerate(roi_rects):
                    roi_data.append({"rect": [rect["x"], rect["y"], rect["w"], rect["h"]], "label": f"ROI_{i}"})
                # 用 sample 圖片縮放成 1080p 作為 ref_img.bmp
                ref_img = cv2.imread(sample_img_path)
                if ref_img is not None:
                    target_height = 1080
                    scale_1080p = target_height / ref_img.shape[0]
                    new_size_1080p = (int(ref_img.shape[1] * scale_1080p), target_height)
                    ref_img_1080p = cv2.resize(ref_img, new_size_1080p, interpolation=cv2.INTER_AREA)
                    # 顯示 1080p 圖片加上 ROI 樣子
                    img_1080p_roi_preview = ref_img_1080p.copy()
                    for i, rect in enumerate(roi_rects):
                        x_1080p = int(rect["x"] * scale_1080p)
                        y_1080p = int(rect["y"] * scale_1080p)
                        w_1080p = int(rect["w"] * scale_1080p)
                        h_1080p = int(rect["h"] * scale_1080p)
                        cv2.rectangle(img_1080p_roi_preview, (x_1080p, y_1080p), (x_1080p + w_1080p, y_1080p + h_1080p), (0, 255, 0), 2)
                        cv2.putText(img_1080p_roi_preview, f"ROI_{i}", (x_1080p, y_1080p - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(f"1080p ROI 預覽 {args.item_verify}", img_1080p_roi_preview)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f"1080p ROI 預覽 {args.item_verify}")
                    assets_dir = os.path.join(os.getcwd(), "SimilarityCheckerLive", "assets")
                    os.makedirs(assets_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(assets_dir, "ref_img.bmp"), ref_img_1080p)
                    # 生成縮放後的 ROI 座標，覆蓋 roi_data
                    roi_data_scaled = []
                    for i, rect in enumerate(roi_rects):
                        x_1080p = int(rect["x"] * scale_1080p)
                        y_1080p = int(rect["y"] * scale_1080p)
                        w_1080p = int(rect["w"] * scale_1080p)
                        h_1080p = int(rect["h"] * scale_1080p)
                        roi_data_scaled.append({"rect": [x_1080p, y_1080p, w_1080p, h_1080p], "label": f"ROI_{i}"})
                    roi_data = roi_data_scaled
                # 執行 use_roi 函數
                #這邊
                result, snapshot = roi_helper.use_roi(
                    device_sn=str(args.item_verify),
                    test_time=args.test_time,
                    roi_data=roi_data,
                    score_func=calculate_image_similarity_by_ssim,
                    score_threshold=similarity_threshold,
                    find_pattern=False,
                    is_debug=args.debug,
                    ref_image=ref_img_1080p
                )
                logging.info(f"Item-verify ROI result: {result}")
                # 顯示比對結果
                # cv2.imshow(f"ItemVerify Result {args.item_verify}", snapshot)
                # cv2.waitKey(0)
                cv2.destroyWindow(f"ItemVerify Result {args.item_verify}")
        else:
            logging.error("Please specify --set-roi, --use-roi, or --item-verify parameter")

    except Exception as e:
        logging.error(f"Program execution error: {e}", exc_info=True)
    finally:
        # Clean up resources
        camera_manager.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
