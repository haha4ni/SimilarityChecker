#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import json
import os
import time
from datetime import datetime
from pygrabber.dshow_graph import FilterGraph


class CameraManager:
    _current_focus_value = -1

    def __init__(self, logger, camera_index, config_file="camera_settings.json",
                 frame_width=1280, frame_height=720):
        self.logger = logger
        self.camera_index = camera_index
        self.config_file = config_file
        self.cap = None

        # Initial parameters for auto mode
        self.auto_params = {
            'frame_width': frame_width,
            'frame_height': frame_height,
            'fps': 30,
            'fourcc': 'MJPG',
            'auto_exposure': 0.75,      # Auto exposure
            'auto_white_balance': 1,    # Auto white balance
            'autofocus': 1,             # Auto focus
        }

    def _list_available_cameras(self, max_test=5):
        graph = FilterGraph()
        devices = graph.get_input_devices()
        print("Detected Cameras:")
        print("=" * 20)
        for i, name in enumerate(devices):
            print(f"Camera[{i}]: {name}")

    def open_camera_with_auto_settings(self):
        """
        Open camera with auto settings (for set-roi mode)
        """
        self.logger.info("Opening camera (auto settings mode)...")

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            self.logger.error(f"Camera ID[{self.camera_index}] is invalid.")
            self._list_available_cameras()
            return

        # Set basic parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.auto_params['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.auto_params['frame_height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.auto_params['fps'])

        # Set codec
        fourcc = cv2.VideoWriter_fourcc(*self.auto_params['fourcc'])
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Enable auto modes
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.auto_params['auto_exposure'])
        self.cap.set(cv2.CAP_PROP_AUTO_WB, self.auto_params['auto_white_balance'])
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, self.auto_params['autofocus'])

        self.logger.info("Camera opened (auto settings mode)")

        # Wait for auto adjustment to stabilize
        time.sleep(3)

        # Clear buffer
        for _ in range(5):
            self.cap.read()

        return self.cap

    def save_current_camera_settings(self):
        """
        Save current camera settings (called when taking photo in set-roi mode)
        """
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Camera not opened, cannot save settings")
            return False

        try:
            # Get all current parameters
            current_settings = {
                'frame_width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                'frame_height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'fourcc': 'MJPG',

                # Switch to manual mode and save current values
                'auto_exposure': 0.25,  # Switch to manual exposure
                'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'hue': self.cap.get(cv2.CAP_PROP_HUE),
                'gamma': self.cap.get(cv2.CAP_PROP_GAMMA),

                'auto_white_balance': 0,  # Switch to manual white balance
                'white_balance_temp': self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE),

                'autofocus': 0,  # Switch to manual focus
                'focus': (self._current_focus_value if getattr(self, '_current_focus_value', -1) >= 0 else self.cap.get(cv2.CAP_PROP_FOCUS)),

                'gain': self.cap.get(cv2.CAP_PROP_GAIN),
                'sharpness': self.cap.get(cv2.CAP_PROP_SHARPNESS),
                'backlight': self.cap.get(cv2.CAP_PROP_BACKLIGHT),

                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(current_settings, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Camera settings saved to: {self.config_file}")
            self.logger.info("Next use-roi mode will use these fixed settings")

            return True

        except Exception as e:
            self.logger.error(f"Error saving camera settings: {e}")
            return False

    def open_camera_with_saved_settings(self):
        """
        Open camera and apply saved settings (for use-roi mode)
        """
        if not os.path.exists(self.config_file):
            self.logger.warning(f"Camera config file {self.config_file} not found")
            self.logger.warning("Will use default settings, recommend running set-roi mode first to save settings")
            return self.open_camera_with_auto_settings()

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            self.logger.info("Loading saved camera settings...")
            self.logger.info(f"Settings saved at: {settings.get('saved_at', 'Unknown')}")

            # Open camera
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")

            # Apply all settings
            success_count = 0
            total_count = 0

            param_mapping = {
                'frame_width': cv2.CAP_PROP_FRAME_WIDTH,
                'frame_height': cv2.CAP_PROP_FRAME_HEIGHT,
                'fps': cv2.CAP_PROP_FPS,
                'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
                'exposure': cv2.CAP_PROP_EXPOSURE,
                'brightness': cv2.CAP_PROP_BRIGHTNESS,
                'contrast': cv2.CAP_PROP_CONTRAST,
                'saturation': cv2.CAP_PROP_SATURATION,
                'hue': cv2.CAP_PROP_HUE,
                'gamma': cv2.CAP_PROP_GAMMA,
                'auto_white_balance': cv2.CAP_PROP_AUTO_WB,
                'white_balance_temp': cv2.CAP_PROP_WB_TEMPERATURE,
                'autofocus': cv2.CAP_PROP_AUTOFOCUS,
                'focus': cv2.CAP_PROP_FOCUS,
                'gain': cv2.CAP_PROP_GAIN,
                'sharpness': cv2.CAP_PROP_SHARPNESS,
                'backlight': cv2.CAP_PROP_BACKLIGHT,
            }

            # Set codec first
            if 'fourcc' in settings:
                fourcc = cv2.VideoWriter_fourcc(*settings['fourcc'])
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # Apply other parameters
            for param_name, cv2_prop in param_mapping.items():
                if param_name in settings:
                    value = settings[param_name]
                    total_count += 1
                    if self.cap.set(cv2_prop, value):
                        self.logger.debug(f"✓ {param_name}: {value}")
                        success_count += 1
                    else:
                        self.logger.warning(f"✗ {param_name}: {value} (failed to set)")

            # Wait for stabilization
            self.logger.info("Waiting for camera to stabilize...")
            time.sleep(2)

            # Clear buffer
            for _ in range(5):
                self.cap.read()

            self.logger.info(f"Camera opened with saved settings ({success_count}/{total_count} parameters)")
            return self.cap

        except Exception as e:
            self.logger.error(f"Error loading camera settings: {e}")
            self.logger.warning("Will use default settings")
            return self.open_camera_with_auto_settings()
        
    def enable_autofocus(self):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            return True
        return False

    def set_manual_focus(self, value):
        """
        The focus value must be mutiples of 5
        """
        if not (self.cap and self.cap.isOpened()):
            return False

        self._current_focus_value = value*5
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ok = self.cap.set(cv2.CAP_PROP_FOCUS, self._current_focus_value)

        return ok

    def release(self):
        """
        Release camera resources
        """
        if self.cap:
            self.cap.release()
