#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import csv
import json
import numpy as np
import os
import sys
from pathlib import Path
from sfr import SFR

PROJECT_NAME = "Gimli"
SFR_TEST_ID = "01"
MTF_PASS_CRITERIA_00F = 0.4
MTF_PASS_CRITERIA_70F = 0.2

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

def calculate_SFR_MTF(img_gray, rect, edge_orientaion, frequency=0.25, gamma=0.5, oversample=4):
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

def process_image(img_path, outdir, n, min_area=2000, max_squares=5):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    img_name = f"{SFR_TEST_ID}-MTF-{Path(img_path).suffix.lstrip('.')}"

    vis = cv2.imread(str(img_path), cv2.IMREAD_COLOR_BGR)
    compression_params = [cv2.IMWRITE_JPEG_QUALITY, 98]
    cv2.imwrite(f"{outdir}/{img_name}.jpg", vis, compression_params)
    img_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{outdir}/{img_name}-yuv.bmp", img_gray)
    if img_gray is None:
        raise FileNotFoundError(img_path)
    h, w = img_gray.shape[:2]

    # Detect black squares
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th_open = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    cnts, _ = cv2.findContours(th_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > w/2 or bh > h/2:
            continue
        ar = bw/float(bh)
        if not (0.5 < ar < 1.8):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cands.append({"contour": c, "bbox": (x, y, bw, bh), "center": (cx, cy), "area": area})

    cands = sorted(cands, key=lambda d: d["area"], reverse=True)[:max_squares]

    # Classify TL/TR/BL/BR/C
    img_cx, img_cy = w//2, h//2
    for d in cands:
        cx, cy = d["center"]
        d["dist2center"] = (cx-img_cx)**2 + (cy-img_cy)**2
    center_idx = np.argmin([d["dist2center"] for d in cands]) if cands else -1
    for i,d in enumerate(cands):
        if i == center_idx:
            d["pos"] = "C"
            d["field"] = "00F"
        else:
            cx, cy = d["center"]
            d["pos"] = ("T" if cy < img_cy else "B") + ("L" if cx < img_cx else "R")
            d["field"] = "70F"

    roi_rule = {
        "TL":("top", "left"),
        "TR":("top", "right"),
        "BL":("bottom", "left"),
        "BR":("bottom", "right"),
        "C":("bottom", "right")
    }

    rois=[]
    for d in cands:
        tag = d["pos"]
        field = d["field"]
        mids = edge_midpoints_from_minarearect(d["contour"])
        for edge_orientation, side in zip(("H", "V"), roi_rule[tag]):
            mx, my = mids[side]
            rx, ry = int(round(mx-n/2)), int(round(my-n/2))
            rx, ry = clamp_roi(rx, ry, n, w, h)
            rect = (rx, ry, n, n)
            sfr_val, edge_angle = calculate_SFR_MTF(img_gray, rect, edge_orientation, frequency=0.25)
            rois.append({
                "which":tag,
                "test_field":field,
                "edge_orientation":edge_orientation,
                "center":(round(mx), round(my)),
                "SFR_MTF":sfr_val,
                "SFR_orientation": "H" if edge_orientation == "V" else "V",
                "edge_angle":edge_angle
            })
            cv2.rectangle(vis, (rx, ry), (rx+n, ry+n), (0, 0, 255), 2)
            cv2.putText(vis, f"{sfr_val:.3f}", (rx+2, ry-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    for d in cands:
        cx,cy = d["center"]
        cv2.putText(vis, d["pos"], (cx-15, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(f"{outdir}/result-{img_name}.jpg", vis)

    output_cvs_center_coordinates(rois, outdir, img_name)
    output_cvs_MTF(rois, outdir, img_name)
    output_json_MTF(rois, outdir)

    json_path = str(outdir/f"test_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"image":Path(img_path).name, "roi_size":n, "rois":rois},
                  f, ensure_ascii=False, indent=2)

    return json_path

######################
# Output Test Result #
######################
def output_cvs_center_coordinates(roi_data, output_folder, image_name):
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

        if field == "00F" and mtf < MTF_PASS_CRITERIA_00F:
            result = "FAILED"
        elif field == "70F" and mtf < MTF_PASS_CRITERIA_70F:
            result = "FAILED"

        key = f"{field}_{pos}_{SFR_orientation}"
        mtfs[key] = mtf * 100

    json_path = str(output_folder/"sfr_test.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "error_code":result,
            "return_values":mtfs
        }, f, ensure_ascii=False, indent=2)

#############
# Main App  #
#############
def main():
    args = parse_arguments()

    process_image(args.test_image, args.outdir, args.roi_size)

if __name__ == "__main__":
    main()
