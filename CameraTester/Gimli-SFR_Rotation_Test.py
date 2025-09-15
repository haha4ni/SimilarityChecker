#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import csv
import json
import math
import numpy as np
import os
import sys
from pathlib import Path
from sfr import SFR

PROJECT_NAME = "Gimli"
SFR_TEST_NAME = "01-SFR"
SFR_MTF_PASS_CRITERIA_00F = 0.4
SFR_MTF_PASS_CRITERIA_70F = 0.2

ROTATION_TEST_NAME = "02-Rotation"
ROTATION_DEGREE_PASS_CRITERIA = 2.0

##############################
# Config & Argument Parsing #
##############################
def parse_arguments():
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} SFR Testing")
    parser.add_argument("-ti", "--test-image", help="Path to the test image")
    parser.add_argument("-o","--outdir", default="roi_out", help="Output directory")
    parser.add_argument("-n","--roi-size", type=int, default=120, help="ROI size (pixels)")
    return parser.parse_args()

####################
# Estimate SFR MTF #
####################
def calculate_SFR_MTF(img, rect, edge_orientaion, frequency=0.25, gamma=0.5, oversample=4):
    x, y, w, h = rect
    if (edge_orientaion == "V"):
        h = h + 1
    else:
        w = w +1
    # ROI (left, upper, right, lower)
    pil_roi = (x, y, x + w, y + h)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = SFR(img_gray, pil_roi).calculate(export_csv=None)
    mtf_curve = out['Channels']['L']['MTF']
    edge_angle = out['Channels']['L']['Edge Angle']
    cy_per_px = out.get("Cy/Pxl", None)
    if cy_per_px is None:
        cy_per_px = np.linspace(0.0, 1.0, len(mtf_curve))
    mtf = float(np.interp(frequency, cy_per_px, mtf_curve))
    return mtf, edge_angle

def calculate_SFR_Frequency(img_gray, rect, edge_orientaion, MTF_target=0.30, gamma=0.5, oversample=4):
    x, y, w, h = rect
    if (edge_orientaion == "V"):
        h = h + 1
    else:
        w = w +1
    # ROI (left, upper, right, lower)
    pil_roi = (x, y, x + w, y + h)
    out = SFR(img_gray, pil_roi).calculate(export_csv=None)
    mtf_curve = out['Channels']['L']['MTF']
    edge_angle = out['Channels']['L']['Edge Angle']
    cy_per_px = out.get("Cy/Pxl", None)
    if cy_per_px is None:
        cy_per_px = np.linspace(0.0, 1.0, len(mtf_curve))
    mtf_array = np.array(mtf_curve)
    cy_array = np.array(cy_per_px)
    frequency = float(np.interp(MTF_target, mtf_array[::-1], cy_array[::-1]))
    return frequency, edge_angle

###############################
# Estimate Pan, Tilt and Roll #
###############################
def build_homography(img_points):
    world = np.array([[-1,-1],[1,-1],[-1,1],[1,1]], dtype=np.float64)
    image = np.array([img_points["TL"], img_points["TR"], img_points["BL"], img_points["BR"]], dtype=np.float64)
    H, _ = cv2.findHomography(world, image, method=0)
    return H

def decompose_pose_from_H(H, cx, cy, f_min=800, f_max=6000):
    # Given principal point (cx,cy) and unknown f, find the most orthogonal R
    # Return Rotation matrix, estimated focal
    def cost_from_f(f):
        K = np.array([[f,0,cx],[0,f,cy],[0,0,1.0]], dtype=np.float64)
        Kinv = np.linalg.inv(K)
        h1, h2, h3 = H[:,0], H[:,1], H[:,2]
        r1 = Kinv @ h1; r2 = Kinv @ h2
        lam = 1.0/np.linalg.norm(r1)
        r1 *= lam; r2 *= lam; r3 = np.cross(r1, r2)
        R = np.column_stack([r1,r2,r3])
        U,S,Vt = np.linalg.svd(R); R = U @ Vt
        c = abs(np.dot(R[:,0],R[:,1])) + abs(np.linalg.norm(R[:,0])-1) + abs(np.linalg.norm(R[:,1])-1)
        if np.linalg.det(R) < 0: c += 1.0
        return c, R
    best = (1e9, None, None)
    for f in np.linspace(f_min, f_max, 200):
        c, R = cost_from_f(f)
        if c < best[0]: best = (c,R,f)
    return best[1], best[2]

def get_rotation_degree(R, flip_pan=False, flip_tilt=False, flip_roll=True):
    # Return (tiltV, panH, rollR) in degrees
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy < 1e-8:
        yaw = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        roll = 0.0
    else:
        yaw   = math.atan2(R[1,0], R[0,0])   # Pan (H)
        pitch = math.atan2(-R[2,0], sy)      # Tilt (V)
        roll  = math.atan2(R[2,1], R[2,2])   # Roll (R)
    V, H_, R_ = np.degrees(pitch), np.degrees(yaw), np.degrees(roll)
    if flip_pan:  H_ = -H_
    if flip_tilt: V  = -V
    if flip_roll: R_ = -R_
    return float(V), float(H_), float(R_)

####################
# Image Processing #
####################
def clamp_roi(x, y, n, w, h):
    x = max(0, min(w - n, x))
    y = max(0, min(h - n, y))
    return x, y

def edge_midpoints_from_minarearect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    mids = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i+1) % 4]
        mids.append(((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0))
    mids_np = np.array(mids)
    return {
        "top": tuple(mids[np.argmin(mids_np[:, 1])]),
        "bottom": tuple(mids[np.argmax(mids_np[:, 1])]),
        "left": tuple(mids[np.argmin(mids_np[:, 0])]),
        "right": tuple(mids[np.argmax(mids_np[:, 0])]),
        "box_points": box.tolist()
    }

def detect_square_centers(img, min_area=700, max_area_ratio=0.08):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 3)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area_ratio*w*h:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03*peri, True)
        if len(approx) < 4:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        if rw < 10 or rh < 10:
            continue
        ar = rw / (rh + 1e-8)
        if ar < 0.5 or ar > 1.6:  # approximately square
            continue
        cands.append({"contour": c, "center": (float(cx), float(cy)), "area": float(area)})
    return cands

def pick_C_TL_TR_BL_BR(cands, w, h, return_tags_order=False):
    """
    Pick C/TL/TR/BL/BR from cands.
    Return (chosen_points, chosen_contours[, tags]):
      - chosen_points  : {"C": (x,y), "TL": (x,y), ...}
      - chosen_contours: {"C": contour, "TL": contour, ...}
      - tags (Optional): ["C","TL","TR","BL","BR"]
    """
    assert len(cands) >= 5, f"need >=5 candidates, got {len(cands)}"

    centers = [tuple(map(float, d["center"])) for d in cands]
    gx = sum(p[0] for p in centers) / len(centers)
    gy = sum(p[1] for p in centers) / len(centers)

    def d2_cand(i, q):
        cx, cy = cands[i]["center"]
        return (cx - q[0])**2 + (cy - q[1])**2

    # 1) Choose C: the nearest point to center
    idxC = min(range(len(cands)), key=lambda i: d2_cand(i, (gx, gy)))

    # 2) Group by quadrant
    quads = {"TL": [], "TR": [], "BL": [], "BR": []}
    for i, d in enumerate(cands):
        cx, cy = d["center"]
        if cy < gy and cx < gx:   quads["TL"].append(i)
        if cy < gy and cx >= gx:  quads["TR"].append(i)
        if cy >= gy and cx < gx:  quads["BL"].append(i)
        if cy >= gy and cx >= gx: quads["BR"].append(i)

    # 3) The corners: choose the nearest point to corner in each quadrant
    corners_img = {
        "TL": (0.0, 0.0),
        "TR": (float(w), 0.0),
        "BL": (0.0, float(h)),
        "BR": (float(w), float(h)),
    }
    idx_map = {"C": idxC}
    used = {idxC}
    for tag in ["TL", "TR", "BL", "BR"]:
        cand_idx = quads[tag] if len(quads[tag]) > 0 else list(range(len(cands)))
        cand_idx = [i for i in cand_idx if i not in used]
        if not cand_idx:
            cand_idx = [i for i in range(len(cands)) if i not in used]
        tgt = corners_img[tag]
        best_i = min(cand_idx, key=lambda i: d2_cand(i, tgt))
        idx_map[tag] = best_i
        used.add(best_i)

    # 4) Use tag as key and ouput dict
    chosen_points = {tag: tuple(map(float, cands[idx_map[tag]]["center"]))
                     for tag in ["C","TL","TR","BL","BR"]}
    chosen_contours = {tag: cands[idx_map[tag]]["contour"]
                       for tag in ["C","TL","TR","BL","BR"]}

    if return_tags_order:
        return chosen_points, chosen_contours, ["C","TL","TR","BL","BR"]
    return chosen_points, chosen_contours

def process_image(img_path, outdir, n, min_area=2000, max_squares=5):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sfr_name_prefix = f"{SFR_TEST_NAME}-{Path(img_path).suffix.lstrip('.')}"
    rotation_name_prefix = f"{ROTATION_TEST_NAME}-{Path(img_path).suffix.lstrip('.')}"

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR_BGR)
    h, w = img_bgr.shape[:2]

    cands = detect_square_centers(img_bgr)
    if len(cands) < 5:
        raise SystemExit(f"Detected {len(cands)} candidate squares (<5), please adjust min_area or image quality.")

    chosen_points, chosen_contours = pick_C_TL_TR_BL_BR(cands, w, h)

    # SFR test
    roi_position = {
        "TL":("top", "left"),
        "TR":("top", "right"),
        "BL":("bottom", "left"),
        "BR":("bottom", "right"),
        "C":("bottom", "right")
    }

    img_sfr_result = img_bgr.copy()
    rois=[]
    for tag, pt in chosen_points.items():
        field = "00F" if tag == "C" else "70F"
        mids = edge_midpoints_from_minarearect(chosen_contours[tag])
        for edge_orientation, side in zip(("H", "V"), roi_position[tag]):
            mx, my = mids[side]
            rx, ry = int(round(mx-n/2)), int(round(my-n/2))
            rx, ry = clamp_roi(rx, ry, n, w, h)
            rect = (rx, ry, n, n)
            sfr_val, edge_angle = calculate_SFR_MTF(img_bgr, rect, edge_orientation, frequency=0.25)
            rois.append({
                "which":tag,
                "test_field":field,
                "edge_orientation":edge_orientation,
                "center":(round(mx), round(my)),
                "SFR_MTF":sfr_val,
                "SFR_orientation": "H" if edge_orientation == "V" else "V",
                "edge_angle":edge_angle
            })
            cv2.rectangle(img_sfr_result, (rx, ry), (rx+n, ry+n), (0, 0, 255), 2)
            cv2.putText(img_sfr_result, f"{sfr_val:.3f}", (rx+2, ry-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    for tag, pt in chosen_points.items():
        cx,cy = pt
        x = round(cx) - 15
        y = round(cy) - 10
        cv2.putText(img_sfr_result, tag, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(f"{outdir}/result-{sfr_name_prefix}.jpg", img_sfr_result)

    # Rotation test
    img_rotation_result = img_bgr.copy()

    H = build_homography(chosen_points)
    cx, cy = chosen_points["C"]
    R, est_f = decompose_pose_from_H(H, cx, cy)
    tilt_v, pan_h, roll_r = get_rotation_degree(R)

    txt = f"Rotation Info(deg): Pan(H)={pan_h:.3f} Tilt(V)={tilt_v:.3f} Roll(R)={roll_r:.3f}"
    cv2.putText(img_rotation_result, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    for tag,p in chosen_points.items():
        col = (0,0,255) if tag=="C" else (255,0,255)
        cv2.circle(img_rotation_result, (int(p[0]), int(p[1])), 10, col, 2)
        cv2.putText(img_rotation_result, tag, (int(p[0])+8, int(p[1])-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imwrite(f"{outdir}/result-{rotation_name_prefix}.jpg", img_rotation_result)

    # Save CVS and JSON
    output_cvs_roi_center_coordinates(rois, outdir, sfr_name_prefix)
    output_cvs_MTF(rois, outdir, sfr_name_prefix)
    output_json_MTF(rois, outdir)
    output_cvs_center_coordinates(chosen_points, outdir, rotation_name_prefix)
    output_cvs_rotation(pan_h, tilt_v, roll_r, outdir, rotation_name_prefix)
    output_json_rotation(pan_h, tilt_v, roll_r, outdir)

    json_path = str(outdir/f"test_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image":Path(img_path).name,
            "contours": chosen_points,
            "rotation": {
                "Pan(H)_deg": pan_h,
                "Tilt(V)_deg": tilt_v,
                "Roll(R)_deg": roll_r
            },
            "roi_size":n,
            "rois":rois},f, ensure_ascii=False, indent=2)

    return json_path

######################
# Output Test Result #
######################
def output_cvs_roi_center_coordinates(roi_data, output_folder, image_name):
    row = {
        "C_H_X": None, "C_H_Y": None, "C_V_X": None, "C_V_Y": None,
        "TL_H_X": None, "TL_H_Y": None, "TL_V_X": None, "TL_V_Y": None,
        "TR_H_X": None, "TR_H_Y": None, "TR_V_X": None, "TR_V_Y": None,
        "BL_H_X": None, "BL_H_Y": None, "BL_V_X": None, "BL_V_Y": None,
        "BR_H_X": None, "BR_H_Y": None, "BR_V_X": None, "BR_V_Y": None,
    }

    for roi in roi_data:
        pos = roi["which"]
        SFR_orientation = roi["SFR_orientation"]
        cx, cy = roi["center"]

        row[f"{pos}_{SFR_orientation}_X"] = cx
        row[f"{pos}_{SFR_orientation}_Y"] = cy

    out_csv = f"{output_folder}/sfrRoiCoordinates-{image_name}.csv"
    fieldnames = list(row.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

def output_cvs_MTF(roi_data, output_folder, image_name):
    row = {
        "00F_C_H": None, "00F_C_V": None,
        "70F_TL_H": None, "70F_TL_V": None,
        "70F_TR_H": None, "70F_TR_V": None,
        "70F_BL_H": None, "70F_BL_V": None,
        "70F_BR_H": None, "70F_BR_V": None,
    }

    for roi in roi_data:
        field = roi["test_field"]
        pos = roi["which"]
        SFR_orientation = roi["SFR_orientation"]
        mtf = round(roi["SFR_MTF"] * 100, 3)

        row[f"{field}_{pos}_{SFR_orientation}"] = mtf

    out_csv = f"{output_folder}/sfr-{image_name}.csv"
    fieldnames = list(row.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

def output_json_MTF(roi_data, output_folder):
    mtfs = {}
    result = "SUCCESS"
    for roi in roi_data:
        field = roi["test_field"]
        pos = roi["which"]
        SFR_orientation = roi["SFR_orientation"]
        mtf = roi["SFR_MTF"]

        if field == "00F" and mtf < SFR_MTF_PASS_CRITERIA_00F:
            result = "FAILED"
        elif field == "70F" and mtf < SFR_MTF_PASS_CRITERIA_70F:
            result = "FAILED"

        key = f"{field}_{pos}_{SFR_orientation}"
        mtfs[key] = mtf * 100

    json_path = str(output_folder/"sfr_test_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "error_code":result,
            "return_values":mtfs
        }, f, ensure_ascii=False, indent=2)

def output_cvs_center_coordinates(points_data, output_folder, image_name):
    row = {
        "C_X": None, "C_Y": None,
        "TL_X": None, "TL_Y": None,
        "TR_X": None, "TR_Y": None,
        "BL_X": None, "BL_Y": None,
        "BR_X": None, "BR_Y": None,
    }

    for k in ["TL","TR","BL","BR","C"]:
        cx, cy = points_data[k]

        row[f"{k}_X"] = round(cx)
        row[f"{k}_Y"] = round(cy)

    out_csv = f"{output_folder}/centerCoordinates-{image_name}.csv"
    fieldnames = list(row.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

def output_cvs_rotation(pan, tilt, roll, output_folder, image_name):
    row = {
        "Pan(V)": None, "Tilt(H)": None, "Roll(R)": None,
    }

    row["Pan(V)"] = round(pan, 3)
    row["Tilt(H)"] = round(tilt, 3)
    row["Roll(R)"] = round(roll, 3)

    out_csv = f"{output_folder}/degree-{image_name}.csv"
    fieldnames = list(row.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

def output_json_rotation(pan, tilt, roll, output_folder):
    degs = {}
    result = "SUCCESS"

    degs["Pan(V)"] = pan
    degs["Tilt(H)"] = tilt
    degs["Roll(R)"] = roll

    if pan >= ROTATION_DEGREE_PASS_CRITERIA or tilt >= ROTATION_DEGREE_PASS_CRITERIA or roll >= ROTATION_DEGREE_PASS_CRITERIA:
        result = "FAILED"

    json_path = str(output_folder/"rotation_degree_test_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "error_code":result,
            "return_values":degs
        }, f, ensure_ascii=False, indent=2)

#############
# Main App  #
#############
def main():
    args = parse_arguments()

    process_image(args.test_image, args.outdir, args.roi_size)

if __name__ == "__main__":
    main()
