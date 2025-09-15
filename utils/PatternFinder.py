#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Tuple, List, Optional
import cv2
import numpy as np


class PatternFinder:
    """PatternFinder using feature matching and homography transformation"""

    def __init__(self, use_orb: bool = False, ratio: float = 0.75, 
                 ransac_reproj: float = 3.0):
        """
        Initialize detector

        Args:
            use_orb: Whether to use ORB (otherwise prefer SIFT)
            ratio: Lowe ratio test threshold (0.7~0.8 commonly used)
            ransac_reproj: RANSAC reprojection threshold (px)
        """
        self.use_orb = use_orb
        self.ratio = ratio
        self.ransac_reproj = ransac_reproj

        # Create feature detector and matcher
        self.feature, self.matcher, self.norm_type, self.is_binary = self._create_detector()

    def _create_detector(self):
        """
        Prefer SIFT (requires opencv-contrib-python). Use ORB if unavailable or specified.
        Returns (feature, matcher, norm_type, is_binary_desc)
        """
        if not self.use_orb and hasattr(cv2, "SIFT_create"):
            sift = cv2.SIFT_create()  # Can add params: nfeatures, contrastThreshold, edgeThreshold, sigma
            # SIFT/AKAZE use L2, FLANN performs well with SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=64)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            return sift, matcher, cv2.NORM_L2, False
        else:
            orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            return orb, matcher, cv2.NORM_HAMMING, True

    def _detect_and_describe(self, img_gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors"""
        kpts, desc = self.feature.detectAndCompute(img_gray, None)
        if desc is None:
            desc = np.zeros((0, 32), dtype=np.uint8)
        return kpts, desc

    def _ratio_test_knn(self, knn_matches) -> List[cv2.DMatch]:
        """Lowe's ratio test to filter good matches"""
        good = []
        for m in knn_matches:
            if len(m) >= 2 and m[0].distance < self.ratio * m[1].distance:
                good.append(m[0])
        return good

    def _estimate_homography(self, kpts_obj: List[cv2.KeyPoint], 
                           kpts_scene: List[cv2.KeyPoint], 
                           matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate homography matrix"""
        if len(matches) < 8:
            return None, None

        src = np.float32([kpts_obj[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kpts_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_reproj)
        return H, mask

    def _draw_detection(self, scene_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Draw detection results on scene image"""
        out = scene_bgr.copy()
        quad = quad.astype(np.int32)
        cv2.polylines(out, [quad], isClosed=True, color=(0, 255, 0), thickness=3)

        # Minimum bounding rotated rectangle for visualization reference
        rr = cv2.minAreaRect(quad.reshape(-1, 2).astype(np.float32))
        box = cv2.boxPoints(rr)
        box = np.int32(box)
        cv2.polylines(out, [box], True, (255, 0, 0), 2)
        return out

    def check_pattern_acceptablility(self, obj_mat: np.ndarray):
         # Validate inputs
        if obj_mat is None:
            raise ValueError("Input images cannot be None")

        if len(obj_mat.shape) == 0:
            raise ValueError("Input images must be valid numpy arrays")

        # Convert to grayscale if needed
        if len(obj_mat.shape) == 3:
            obj = cv2.cvtColor(obj_mat, cv2.COLOR_BGR2GRAY)
        else:
            obj = obj_mat.copy()

        # Feature detection
        kp, desc = self._detect_and_describe(obj)

        isPass = True
        if (len(kp) < 8 or desc is None or len(desc) == 0):
            isPass = False

        return isPass, len(kp), len(desc)

    def detect_logo_rect(self, obj_mat: np.ndarray, scene_mat: np.ndarray) -> Tuple[np.ndarray, List, List, int, int]:
        """
        Detect logo position in scene

        Args:
            obj_mat: Template image (logo) as OpenCV Mat (numpy array)
            scene_mat: Scene image as OpenCV Mat (numpy array)

        Returns:
            tuple: (visualization result, quad coordinates, rotated rect points, good matches count, inliers count)

        Raises:
            ValueError: Invalid input images
            RuntimeError: Insufficient features or cannot estimate homography
        """
        # Validate inputs
        if obj_mat is None or scene_mat is None:
            raise ValueError("Input images cannot be None")

        if len(obj_mat.shape) == 0 or len(scene_mat.shape) == 0:
            raise ValueError("Input images must be valid numpy arrays")

        # Convert to grayscale if needed
        if len(obj_mat.shape) == 3:
            obj = cv2.cvtColor(obj_mat, cv2.COLOR_BGR2GRAY)
        else:
            obj = obj_mat.copy()

        if len(scene_mat.shape) == 3:
            scene_gray = cv2.cvtColor(scene_mat, cv2.COLOR_BGR2GRAY)
            scene_bgr = scene_mat.copy()
        else:
            scene_gray = scene_mat.copy()
            scene_bgr = cv2.cvtColor(scene_mat, cv2.COLOR_GRAY2BGR)

        # Feature detection
        k1, d1 = self._detect_and_describe(obj)
        k2, d2 = self._detect_and_describe(scene_gray)

        if (len(k1) < 8 or len(k2) < 8 or d1 is None or d2 is None or 
            len(d1) == 0 or len(d2) == 0):
            raise RuntimeError("Insufficient features, please use clearer or larger template/scene images.")

        # KNN matching
        # FLANN doesn't support uint8 binary descriptors, need to convert to float32; but ORB uses BFMatcher
        if isinstance(self.matcher, cv2.FlannBasedMatcher) and d1.dtype != np.float32:
            d1 = d1.astype(np.float32)
            d2 = d2.astype(np.float32)

        knn = self.matcher.knnMatch(d1, d2, k=2)
        good = self._ratio_test_knn(knn)

        # Estimate homography
        H, inlier_mask = self._estimate_homography(k1, k2, good)
        if H is None:
            raise RuntimeError("Cannot estimate homography, please adjust ratio or change feature method/images.")

        # Project template corners
        h, w = obj.shape[:2]
        obj_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        scene_corners = cv2.perspectiveTransform(obj_corners, H).reshape(-1, 2)

        # Visualization
        vis = self._draw_detection(scene_bgr, scene_corners)

        # Get rotated rectangle (numerical result)
        rr = cv2.minAreaRect(scene_corners.astype(np.float32))
        box = cv2.boxPoints(rr)  # 4x2
        box = box.tolist()

        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

        return vis, scene_corners.tolist(), box, len(good), n_inliers


def main():
    """Command line interface"""
    ap = argparse.ArgumentParser(description="Find pattern rect via feature matching + homography")
    ap.add_argument("logo", help="Template image (e.g., Coca-Cola logo) path")
    ap.add_argument("scene", help="Scene image path")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio (0.7~0.8 commonly used)")
    ap.add_argument("--reproj", type=float, default=3.0, help="RANSAC reprojection threshold (px)")
    ap.add_argument("--orb", action="store_true", help="Force use ORB (otherwise prefer SIFT)")
    ap.add_argument("--out", default="detected.png", help="Output image path")
    args = ap.parse_args()

    # Create detector and run detection
    detector = PatternFinder(
        use_orb=args.orb,
        ratio=args.ratio,
        ransac_reproj=args.reproj
    )

    logo_img = cv2.imread(args.logo)
    scene_img = cv2.imread(args.scene)

    vis, quad, rot_box, n_good, n_inlier = detector.detect_logo_rect(logo_img, scene_img)

    cv2.imwrite(args.out, vis)
    print(f"[OK] Output saved: {args.out}")
    print(f"Good matches: {n_good}, inliers: {n_inlier}")
    print("Quadrilateral coordinates (logo corners projected to scene):")
    for p in quad:
        print(f"  ({p[0]:.1f}, {p[1]:.1f})")
    print("Minimum bounding rotated rectangle points:")
    for p in rot_box:
        print(f"  ({p[0]:.1f}, {p[1]:.1f})")


if __name__ == "__main__":
    main()
