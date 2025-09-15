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
TEST_NAME = "02-Rotation"
ROTATION_PASS_CRITERIA = 2.0

##############################
# Config & Argument Parsing #
##############################
def parse_arguments():
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} Rotation Degree Testing")
    parser.add_argument("-ti", "--test-image", help="Path to the test image")
    parser.add_argument("-o","--outdir", default="roi_out", help="Output directory")
    parser.add_argument("--flip-pan", action="store_true", help="Flip Pan sign when needed to align with Imatest")
    parser.add_argument("--flip-tilt", action="store_true", help="Flip Tilt sign when needed to align with Imatest")
    parser.add_argument("--flip-roll", action="store_true", help="Flip Roll sign when needed to align with Imatest")
    return parser.parse_args()

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
        cands.append({"center": (float(cx), float(cy)), "area": float(area)})
    return cands

def pick_C_TL_TR_BL_BR(cands, w, h):
    pts = [d["center"] for d in cands]
    gx = sum(p[0] for p in pts)/len(pts)
    gy = sum(p[1] for p in pts)/len(pts)
    def d2(p,q): return (p[0]-q[0])**2 + (p[1]-q[1])**2
    C = min(pts, key=lambda p: d2(p, (gx,gy)))
    quads = {"TL": [], "TR": [], "BL": [], "BR": []}
    for p in pts:
        if p[0] < gx and p[1] < gy: quads["TL"].append(p)
        if p[0] >= gx and p[1] < gy: quads["TR"].append(p)
        if p[0] < gx and p[1] >= gy: quads["BL"].append(p)
        if p[0] >= gx and p[1] >= gy: quads["BR"].append(p)
    corners_img = {"TL": (0.0,0.0), "TR": (float(w),0.0), "BL": (0.0,float(h)), "BR": (float(w), float(h))}
    chosen = {"C": C}
    for tag in ["TL","TR","BL","BR"]:
        cand_list = quads[tag] if len(quads[tag])>0 else pts
        tgt = corners_img[tag]
        chosen[tag] = min(cand_list, key=lambda p: d2(p, tgt))
    return chosen

def process_image(img_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    img_name = f"{TEST_NAME}-{Path(img_path).suffix.lstrip('.')}"

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_BGR)
    h, w = img.shape[:2]

    cands = detect_square_centers(img)
    if len(cands) < 5:
        raise SystemExit(f"Detected {len(cands)} candidate squares (<5), please adjust min_area or image quality.")

    chosen_points = pick_C_TL_TR_BL_BR(cands, w, h)

    # === Build H: world(ideal plane) -> image ===
    H = build_homography(chosen_points)

    # === Estimate pose R, f ===
    cx, cy = chosen_points["C"]
    R, est_f = decompose_pose_from_H(H, cx, cy)

    # === Get angles (can use flip_* to fine-tune sign alignment with Imatest) ===
    tilt_v, pan_h, roll_r = get_rotation_degree(R)

    txt = f"Rotation Info(deg): Pan(H)={pan_h:.3f} Tilt(V)={tilt_v:.3f} Roll(R)={roll_r:.3f}"
    cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    for tag,p in chosen_points.items():
        col = (0,0,255) if tag=="C" else (255,0,255)
        cv2.circle(img, (int(p[0]), int(p[1])), 10, col, 2)
        cv2.putText(img, tag, (int(p[0])+8, int(p[1])-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imwrite(f"{outdir}/result-{img_name}.jpg", img)

    output_cvs_center_coordinates(chosen_points, outdir, img_name)
    output_cvs_rotation(pan_h, tilt_v, roll_r, outdir, img_name)
    output_json_rotation(pan_h, tilt_v, roll_r, outdir)

    json_path = str(outdir/f"{img_name}-test.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image": img_name,
            "focal_px_estimated": est_f,
            "angles_degree": {
                "Pan(H)_deg": pan_h,
                "Tilt(V)_deg": tilt_v,
                "Roll(R)_deg": roll_r
            },
            "points_px": chosen_points
        },f, ensure_ascii=False, indent=2)

    return json_path

######################
# Output Test Result #
######################
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

    if abs(pan) >= ROTATION_PASS_CRITERIA or abs(tilt) >= ROTATION_PASS_CRITERIA or abs(roll) >= ROTATION_PASS_CRITERIA:
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

    process_image(args.test_image, args.outdir)

if __name__ == "__main__":
    main()
