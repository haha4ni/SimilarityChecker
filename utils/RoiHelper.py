#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import inspect
import numpy as np
import threading
import time
from PIL import Image
from typing import Tuple


class RoiHelper:
    SIMILARITY_CHECKER_LIVE_NAME = "SimilarityCheckerLive"
    COLOR_CHECKER_LIVE_NAME = "ColorCheckerLive"

    def __init__(self, logger, camera_manager):
        self.logger = logger
        self.camera_manager = camera_manager
        self.is_loading_camera = False
        self.caller_info = os.path.basename(inspect.stack()[1].filename)

    def _draw_text_with_background(self, img, text, position, font, font_scale, text_color, bg_color, thickness):
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

    def _play_loading_gif(self, window_name, gif_path):
        if not os.path.exists(gif_path):
            return

        gif = Image.open(gif_path)
        frames = []
        try:
            while True:
                frame = gif.convert('RGB')
                frame_np = np.array(frame)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                frame_bgr = cv2.resize(frame_bgr, (150, 150))
                frames.append(frame_bgr)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        idx = 0
        while self.is_loading_camera:
            cv2.imshow(window_name, frames[idx % len(frames)])
            idx += 1
            if cv2.waitKey(50) & 0xFF == 27:
                break
        cv2.destroyWindow(window_name)

    def _select_multiple_rois(self, image) -> np.ndarray:
        def get_label_from_gui(default_label):
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            label = simpledialog.askstring("ROI Label", f"Enter label for ROI (default: {default_label}):")
            root.destroy()
            return label or default_label

        clone = image.copy()
        rois_with_labels = []
        print("[INFO] Select ROI with mouse, press ENTER or SPACE to confirm, ESC to finish.")

        # Create detector with custom parameters
        from utils.PatternFinder import PatternFinder
        pattern_findner = PatternFinder(use_orb=False, ratio=0.75, ransac_reproj=3.0)

        while True:
            roi = cv2.selectROI("Select ROI", clone, fromCenter=False, showCrosshair=False)
            if roi[2] == 0 or roi[3] == 0:
                break
            x, y, w, h = roi
            roi_pass, kp_count, desc_count = pattern_findner.check_pattern_acceptablility(clone[y:y+h, x:x+w])
            if roi_pass:
                default_label = f"ROI_{len(rois_with_labels)}"
                label = get_label_from_gui(default_label)
                rois_with_labels.append({"rect": roi, "label": label})
                cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(clone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                messagebox.showwarning("Warning", f"ROI invalid. Keypoint(={kp_count}) or descriptor(={desc_count}) is not enough.\nPlease select larger area.")
                print(f"[INFO] ROI invalid. Keypoint(={kp_count}) or descriptor(={desc_count}) is not enough.")

        cv2.destroyWindow("Select ROI")
        return rois_with_labels

    def set_roi(self) -> np.ndarray:
        # Show loading animation (if GIF file exists)
        loading_thread = None
        if os.path.exists("./assets/loading.gif"):
            self.is_loading_camera = True
            loading_thread = threading.Thread(target=self._play_loading_gif, args=("Loading", "./assets/loading.gif"))
            loading_thread.start()

        # Open camera with auto settings
        cap = self.camera_manager.open_camera_with_auto_settings()

        if loading_thread:
            self.is_loading_camera = False
            loading_thread.join()

        if not cap:
            exit(-1)

        # Focus UI
        focus_mode = "AUTO"
        focus_value = 25
        trackbar_initialized = False

        window_name = "Set ROI Mode - Camera Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def on_focus_trackbar(val):
            print("on_focus_trackbar")
            nonlocal focus_mode, focus_value, trackbar_initialized
            focus_value = int(val)

            if not trackbar_initialized:
                trackbar_initialized = True
                return
            if focus_mode != "MANUAL":
                focus_mode = "MANUAL"
            ok = self.camera_manager.set_manual_focus(focus_value)
            if not ok:
                self.logger.warning(f"Set manual focus failed at {focus_value}")

        cv2.createTrackbar("Focus Step", window_name, focus_value, 50, on_focus_trackbar)

        self.logger.info("Camera ready, press SPACE to capture for ROI selection")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Add instruction text
            overlay = frame.copy()

            focus_text = f"Focus: {focus_mode}" + (f" ({focus_value*5})" if focus_mode == "MANUAL" else "")
            self._draw_text_with_background(
                overlay, focus_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), (0, 0, 0), 2
            )
            self._draw_text_with_background(
                overlay, "Press SPACE to capture ROI frame",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), (0, 0, 0), 2
            )
            self._draw_text_with_background(
                overlay, "ESC to cancel | Press 'a' to toggle Auto Focus",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), (0, 0, 0), 1
            )

            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                self.logger.info("User cancelled ROI setup")
                break
            elif key == 32:  # SPACE
                roi_image = frame.copy()

                # Save camera settings
                self.logger.info("Saving camera settings...")
                settings_saved = self.camera_manager.save_current_camera_settings()
                if settings_saved:
                    self.logger.info("✓ Camera settings saved")
                else:
                    self.logger.warning("✗ Camera settings save failed")

                # Save reference image
                os.makedirs("assets", exist_ok=True)
                cv2.imwrite("assets/ref_img.bmp", roi_image)
                self.logger.info("Reference image saved to assets/ref_img.bmp")

                cv2.destroyAllWindows()
                return self._select_multiple_rois(roi_image)
            elif key == ord('a'):
                # Switch to auto focus
                if self.camera_manager.enable_autofocus():
                    focus_mode = "AUTO"
                    self.logger.info("Switched to Auto Focus")
                else:
                    self.logger.warning("Failed to switch to Auto Focus")

    def use_roi(self, device_sn, test_time, roi_data, score_func, score_threshold, find_pattern, is_debug) -> Tuple[str, np.ndarray]:
        # Show loading animation (if GIF file exists)
        loading_thread = None
        if os.path.exists("./assets/loading.gif"):
            self.is_loading_camera = True
            loading_thread = threading.Thread(target=self._play_loading_gif, args=("Loading", "./assets/loading.gif"))
            loading_thread.start()

        # Open camera with saved settings
        cap = self.camera_manager.open_camera_with_saved_settings()

        if loading_thread:
            self.is_loading_camera = False
            loading_thread.join()

        if not cap:
            exit(-1)

        ref_img = cv2.imread("assets/ref_img.bmp")
        if ref_img is None:
            self.logger.error("Cannot load reference image assets/ref_img.bmp")
            return

        rois = [(tuple(roi["rect"]), roi["label"]) for roi in roi_data]
        ref_roi_list = [ref_img[y:y+h, x:x+w] for ((x, y, w, h), label) in rois]

        start_time = time.time()
        final_result = "FAIL"

        self.logger.info(f"Starting detection, test time: {test_time} seconds")

        # Create detector with custom parameters
        from utils.PatternFinder import PatternFinder
        pattern_findner = PatternFinder(use_orb=False, ratio=0.75, ransac_reproj=3.0)

        detected_rois = []
        frame_count = 0
        start_fps_time = time.time()
        while time.time() - start_time < test_time or is_debug:
            remaining = test_time - int(time.time() - start_time)
            if is_debug:
                remaining = -1

            ret, frame = cap.read()
            if not ret:
                continue

            # count FPS
            frame_count += 1
            current_time = time.time()

            elapsed = current_time - start_fps_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            test_img = frame
            failed_info = []
            scores = []

            for i, ((x, y, w, h), label) in enumerate(rois):
                ref_roi = ref_roi_list[i]
                if find_pattern and not len(detected_rois) == len(rois):
                    vis, quad, rot_box, n_good, n_inlier = pattern_findner.detect_logo_rect(ref_roi, frame)
                    rot_box_array = np.array(rot_box, dtype=np.float32)
                    box_x, box_y, box_w, box_h = cv2.boundingRect(rot_box_array.astype(np.int32))
                    detected_rois.append((box_x, box_y, box_w, box_h))

                if find_pattern:
                    x, y, w, h = detected_rois[i]

                test_roi = test_img[y:y+h, x:x+w]

                if self.SIMILARITY_CHECKER_LIVE_NAME in self.caller_info:
                    score = score_func(ref_roi, test_roi)
                    scores.append(score)
                elif self.COLOR_CHECKER_LIVE_NAME in self.caller_info:
                    score = score_func(ref_roi, test_roi)
                    avg_hsv = cv2.mean(cv2.cvtColor(test_roi, cv2.COLOR_BGR2HSV))[:3]
                    scores.append((score, avg_hsv))

                if score < score_threshold:
                    failed_info.append((i, label, test_roi))

            overlay = test_img.copy()
            for i, ((x, y, w, h), label) in enumerate(rois):
                if self.SIMILARITY_CHECKER_LIVE_NAME in self.caller_info:
                    score = scores[i]
                elif self.COLOR_CHECKER_LIVE_NAME in self.caller_info:
                    score, avg_hsv = scores[i]

                cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
                color = (0, 255, 0) if i not in [f[0] for f in failed_info] else (0, 0, 255)
                if find_pattern:
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    x, y, w, h = detected_rois[i]

                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)

                # Draw label on top
                self._draw_text_with_background(
                    overlay, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    text_color=color, bg_color=(0, 0, 0), thickness=1
                )

                if self.SIMILARITY_CHECKER_LIVE_NAME in self.caller_info:
                    score_info = f"{score:.2f}"
                elif self.COLOR_CHECKER_LIVE_NAME in self.caller_info:
                    score_info = f"Score: {score:.2f}, HSV: ({avg_hsv[0]:.0f},{avg_hsv[1]:.0f},{avg_hsv[2]:.0f})"

                # Draw score & keypoints below
                self._draw_text_with_background(
                    overlay, score_info, (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    text_color=color, bg_color=(0, 0, 0), thickness=1
                )

            result = "PASS" if not failed_info else "FAIL"
            self._draw_text_with_background(overlay, f"Device SN: {device_sn} | Avg. FPS: {current_fps:.1f} | Time left: {remaining}s",
                                      (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), (0, 0, 0), 2)
            self._draw_text_with_background(overlay, f"Result: {result}",
                                      (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                      (0, 255, 0) if result == "PASS" else (0, 0, 255), (0, 0, 0), 2)
            cv2.imshow("Live CheckMonitor", overlay)

            if result == "PASS" and not is_debug:
                final_result = "PASS"
                break

            key = cv2.waitKey(1) & 0xFF

            if key == 27: # ESC
                break
            elif key == ord('f'): # Find pattern again
                detected_rois = []

        return final_result, overlay

