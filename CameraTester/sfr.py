#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import time

import cv2
import numpy

from scipy import stats
from scipy.fftpack import fft

class SFR():
    image_mat = None
    image_name = ""
    image_roi = (0,0,0,0)
    gamma = 0
    oversampling_rate = 0
    outputs = None

    def __init__(self, image_mat, image_roi, image_name="", gamma=0.5, oversampling_rate=4):
        """
        Initialize SFR analyzer
        
        Parameters:
        - image_mat: OpenCV Mat object (BGR or grayscale format)
        - image_roi: ROI region (left, top, right, bottom)
        - image_name: Image name for output records (optional)
        - gamma: Gamma correction value, default 0.5
        - oversampling_rate: Oversampling rate, default 4
        """
        self.image_mat = image_mat.copy()  # Copy to avoid modifying original image
        self.image_name = image_name if image_name else "opencv_image"
        self.image_roi = image_roi
        self.gamma = gamma
        self.oversampling_rate = oversampling_rate

    def calculate(self, export_csv=None):
        # Get original image size
        if len(self.image_mat.shape) == 3:
            height, width, channels = self.image_mat.shape
        else:
            height, width = self.image_mat.shape
            channels = 1
        
        image_size = (width, height)
        
        # Crop ROI region
        left, top, right, bottom = self.image_roi
        roi_image = self.image_mat[top:bottom, left:right]
        
        # automatically rotate the image if its width is greater than height (clockwise)
        if roi_image.shape[1] > roi_image.shape[0]:
            roi_image = cv2.rotate(roi_image, cv2.ROTATE_90_CLOCKWISE)

        # create dictionary for output data
        outputs = {}
        outputs['File'] = self.image_name
        outputs['Size'] = image_size
        outputs['ROI'] = self.image_roi
        outputs['Gamma'] = self.gamma
        outputs['Oversampling'] = self.oversampling_rate
        outputs['Orientation'] = 'Horizontal' if (self.image_roi[2]-self.image_roi[0]) > (self.image_roi[3]-self.image_roi[1]) else 'Vertical'
        outputs['Run Date'] = time.asctime(time.localtime(time.time()))
        
        # do the same process in R/G/B/L channels
        channels = ['L']
        #channels += ['R', 'G', 'B']  # Enable this line if need to process color channels
        channels_data = {}
        
        for c in channels:
            if c == 'L':
                # Convert to grayscale
                if len(roi_image.shape) == 3:
                    gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = roi_image
                pixel_array = gray_image.astype(numpy.float64) / 255.0  # Normalize to 0-1
            elif c == 'R':
                # In BGR format, R channel is index 2
                pixel_array = roi_image[:, :, 2].astype(numpy.float64) / 255.0
            elif c == 'G':
                # In BGR format, G channel is index 1
                pixel_array = roi_image[:, :, 1].astype(numpy.float64) / 255.0
            elif c == 'B':
                # In BGR format, B channel is index 0
                pixel_array = roi_image[:, :, 0].astype(numpy.float64) / 255.0
            
            # undo gamma correction
            pixel_array = numpy.power(pixel_array, 1/self.gamma)
            
            # get ESF data
            esf_data, slope, intercept = self._get_esf_data(pixel_array, self.oversampling_rate)
            # get LSF data
            lsf_data = self._get_lsf_data(esf_data)
            # get SFR data
            sfr_data = self._get_sfr_data(lsf_data)
            # get MTF data
            mtf_data, mtf50, mtf50p = self._get_mtf_data(sfr_data, self.oversampling_rate)
            # get standardized sharpening MTF data
            standard_mtf_data, standard_mtf50, mtf_equal, k_sharp, sharpening_radius = self._get_standard_mtf_data(mtf_data, mtf50, self.oversampling_rate)
            
            # export results
            if 'Cy/Pxl' not in outputs:
                outputs['Cy/Pxl'] = numpy.linspace(0.0, 1.0, len(mtf_data))
            if 'LW/PH' not in outputs:
                outputs['LW/PH'] = numpy.linspace(0.0, image_size[0 if outputs['Orientation'] == 'Horizontal' else 1]*2, len(mtf_data))
            
            data = {}
            data['ESF'] = esf_data
            data['LSF'] = lsf_data
            data['MTF'] = mtf_data
            data['MTF50'] = mtf50
            data['MTF50P'] = mtf50p
            data['Sharpening'] = (mtf_equal-1)
            data['Corrected MTF'] = standard_mtf_data
            data['Corrected MTF50'] = standard_mtf50
            # NOTE: need to revese the slope since (0, 0) in the image is at the top left corner rahter than at the bottom left corner
            data['Edge Angle'] = numpy.arctan(-slope)*180/numpy.pi
            data['Sharpening Radius'] = sharpening_radius
            channels_data[c] = data
            
        outputs['Channels'] = channels_data
        self.outputs = outputs

        if export_csv:
            self._export_csv_file(export_csv, outputs)

        return outputs

    @staticmethod
    def convert_outputs_to_csv_files(export_csv, outputs):
        """Static method to convert output data to CSV file"""
        sfr_instance = SFR(numpy.zeros((1,1)), (0,0,1,1))  # Create temporary instance
        sfr_instance._export_csv_file(export_csv, outputs)

    def _get_esf_data(self, pixel_array, oversampling_rate):
        edge_idx_per_line = []
        # find edge positions
        for line in pixel_array:
            max_diff = 0
            last_px = line[0]
            max_idx = 0
            for idx, px in enumerate(line):
                diff = abs(last_px - px)
                if diff > max_diff:
                   max_diff = diff
                   max_idx = idx
                last_px = px
            edge_idx_per_line.append(max_idx)
        
        # get line regression result for projection
        slope, intercept, r_value, p_value, std_err = stats.linregress(list(range(len(edge_idx_per_line))), edge_idx_per_line)
        
        # get inspection width
        inspection_width = 1
        # TODO: check if we should remove then "=" condition in the if statement
        while inspection_width <= len(pixel_array[0]):
            inspection_width *= 2
        inspection_width = inspection_width // 2
        half_inspection_width = inspection_width / 2
        
        # do edge spread function
        esf_sum = [0.0] * (inspection_width * oversampling_rate + 2)
        hit_count = [0] * (inspection_width * oversampling_rate + 2)
        
        for y, line in enumerate(pixel_array):
            for x, px in enumerate(line):
                # only calculate the pixels in the inspection width
                if abs(x - (y * slope + intercept)) <= half_inspection_width + 1 / oversampling_rate:
                    idx = int((x - (y * slope + intercept) + half_inspection_width) * oversampling_rate + 1)
                    if 0 <= idx < len(esf_sum):
                        esf_sum[idx] += px
                        hit_count[idx] += 1
        
        # force hit count to 1 if it's 0 to avoid calculation error
        # TODO: we should lower the oversampling rate or shutdown the SFR process if a hit count is 0
        hit_count = [1 if c == 0 else c for c in hit_count]
        return numpy.divide(esf_sum, hit_count).tolist(), slope, intercept

    def _get_lsf_data(self, esf_data):
        # do line spread function
        lsf_data = [0.0] * (len(esf_data) - 2)
        for idx in range(len(lsf_data)):
            # the 3-point derivative
            lsf_data[idx] = (esf_data[idx + 2] - esf_data[idx]) / 2
        return lsf_data

    def _get_sfr_data(self, lsf_data):
        # use hamming window to reduce the effects of the Gibbs phenomenon
        hamming_window = numpy.hamming(len(lsf_data))
        windowed_lsf_data = numpy.multiply(lsf_data, hamming_window)
        raw_sfr_data = numpy.abs(fft(windowed_lsf_data))
        sfr_base = raw_sfr_data[0]
        return [d / sfr_base for d in raw_sfr_data]

    def _get_mtf_data(self, sfr_data, oversampling_rate):
        # When PictureHeight = 2448 & every 2 pixels are corresponding to 1 line pair, then 2448 (LW/PH) = 1224 (LP/PH) = 0.5 (Cy/Pxl) = Nyquist Frequency.
        peak_mtf = 0
        freq_at_50p_mtf = 0
        freq_at_50_mtf = 0
        
        # 1. The SFR series is a symmetry series, so we only need the first half of the series to do the calculation
        # 2. In the original image (oversampling = 1), the valid frequency is 0 ~ 0.5. So the valid frequency after the oversampling is 0 ~ 0.5 * oversmapling rate.
        #    Since we only care about the range from 0 to 1, we should also truncate the series here.
        mtf_data = [0.0] * int(len(sfr_data) / 2 / (oversampling_rate * 0.5))
        
        for idx in range(len(mtf_data)):
            sfr = sfr_data[idx]
            # frequency is from 0 to 1
            freq = idx / (len(mtf_data) - 1) if len(mtf_data) > 1 else 0
            
            # divide by a corrective factor MTF(deriv)
            #     MTF(system) = SFR / FR / MTF(deriv)
            #     MTF(deriv) = sin(PI*f*k*(1/overSamplingFactor)) / (PI*f*k*(1/overSamplingFactor)); k=2 (for the 3-point derivative), k=1 (for the 2-point derivative)
            if freq == 0:
                mtf_data[idx] = sfr
            else:
                correction_factor = (numpy.pi * freq * 2 / oversampling_rate) / numpy.sin(numpy.pi * freq * 2 / oversampling_rate)
                mtf_data[idx] = sfr * correction_factor
            
            # get MTF50
            if freq_at_50_mtf == 0 and mtf_data[idx] < 0.5 and idx > 0:
                freq_at_50_mtf = (idx - 1 + (0.5 - mtf_data[idx]) / (mtf_data[idx - 1] - mtf_data[idx])) / (len(mtf_data) - 1)
            
            # get MTF50P
            if peak_mtf < mtf_data[idx]:
                peak_mtf = mtf_data[idx]
            if freq_at_50p_mtf == 0 and mtf_data[idx] < 0.5 * peak_mtf and idx > 0:
                freq_at_50p_mtf = (idx - 1 + (0.5 * peak_mtf - mtf_data[idx]) / (mtf_data[idx - 1] - mtf_data[idx])) / (len(mtf_data) - 1)
        
        return mtf_data, freq_at_50_mtf, freq_at_50p_mtf

    def _get_standard_mtf_data(self, mtf_data, freq_at_50_mtf, oversampling_rate):
        if freq_at_50_mtf < 0.2:
            freq_equal = 0.6 * freq_at_50_mtf
            sharpening_radius = 3
        else:
            freq_equal = 0.15
            sharpening_radius = 2
        
        idx_equal = freq_equal * (len(mtf_data) - 1)
        int_idx = int(idx_equal)
        if int_idx + 1 < len(mtf_data):
            mtf_equal = mtf_data[int_idx] + (mtf_data[int_idx + 1] - mtf_data[int_idx]) * (idx_equal - int_idx)
        else:
            mtf_equal = mtf_data[int_idx]
        
        last_sharpening_radius = 0
        k_sharp = 0
        
        while last_sharpening_radius != sharpening_radius:
            last_sharpening_radius = sharpening_radius
            # calculate sharpness coefficient
            #     MTF(system) = MTF(standard) * MTF(sharp) = MTF(system) * (1 - ksharp * cos(2*PI*f*R/dscan)) / (1- ksharp)
            #     When MTF(sharp) = 1, ksharp = (1 - MTF(system)) / (cos(2*PI*f*R/dscan) - MTF(system))
            cos_value = numpy.cos(2 * numpy.pi * freq_equal * sharpening_radius)
            if cos_value != mtf_equal:
                k_sharp = (1 - mtf_equal) / (cos_value - mtf_equal)
            else:
                k_sharp = 0
            
            # standardized sharpening
            standard_freq_at_50_mtf = 0
            standard_mtf_data = [0.0] * len(mtf_data)
            
            for idx, mtf in enumerate(mtf_data):
                # frequency is from 0 to 1
                freq = idx / (len(mtf_data) - 1) if len(mtf_data) > 1 else 0
                
                denominator = (1 - k_sharp * numpy.cos(2 * numpy.pi * freq * sharpening_radius)) / (1 - k_sharp)
                if denominator != 0:
                    standard_mtf_data[idx] = mtf / denominator
                else:
                    standard_mtf_data[idx] = mtf
                
                # get MTF50
                if standard_freq_at_50_mtf == 0 and standard_mtf_data[idx] < 0.5 and idx > 0:
                    standard_freq_at_50_mtf = (idx - 1 + (0.5 - standard_mtf_data[idx]) / (standard_mtf_data[idx - 1] - standard_mtf_data[idx])) / (len(standard_mtf_data) - 1)
                    # If the difference of the original frequency at MTF50 and the frequency at MTF50(corr) is larger than 0.04,
                    # it should increase the radius by one and recalculate the ksharp.
                    if abs(standard_freq_at_50_mtf - freq_at_50_mtf) > 0.04:
                        sharpening_radius += 1
                        break
        
        return standard_mtf_data, standard_freq_at_50_mtf, mtf_equal, k_sharp, sharpening_radius

    def _export_csv_file(self, csv_file_name, outputs):
        # TODO: Find a more efficient way to export a CSV file
        with open(csv_file_name, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # export generic data
            writer.writerow(['File', outputs['File']])
            writer.writerow(['Run Date', outputs['Run Date']])
            writer.writerow(['Path', 'OpenCV Mat Input'])
            writer.writerow('')
            # export misc data
            writer.writerow(['Slice Orientation', outputs['Orientation']])
            writer.writerow(['Image WxH pixels', outputs['Size'][0], outputs['Size'][1]])
            writer.writerow(['ROI WxH pixels', outputs['ROI'][2]-outputs['ROI'][0], outputs['ROI'][3]-outputs['ROI'][1]])
            writer.writerow(['ROI boundary L T R B', outputs['ROI'][0], outputs['ROI'][1], outputs['ROI'][2], outputs['ROI'][3]])
            writer.writerow(['Gamma', outputs['Gamma']])
            writer.writerow(['Oversampling', outputs['Oversampling']])
            
            header = ['Channel']
            for c in outputs['Channels'].keys():
                header.append(c)
            writer.writerow(header)
            
            data = ['Edge Angle']
            for c in outputs['Channels'].keys():
                data.append(round(outputs['Channels'][c]['Edge Angle'],4))
            writer.writerow(data)
            
            data = ['MTF50 Cy/pxl (uncorr)']
            for c in outputs['Channels'].keys():
                data.append(round(outputs['Channels'][c]['MTF50'],4))
            writer.writerow(data)
            
            data = ['MTF50 LW/PH (uncorr)']
            for c in outputs['Channels'].keys():
                data.append(round(outputs['Channels'][c]['MTF50']*outputs['Size'][0 if outputs['Orientation'] == 'Horizontal' else 1]*2))
            writer.writerow(data)
            
            data = ['MTF50 Cy/pxl (corr)']
            for c in outputs['Channels'].keys():
                data.append(round(outputs['Channels'][c]['Corrected MTF50'],4))
            writer.writerow(data)
            
            data = ['MTF50 LW/PH (corr)']
            for c in outputs['Channels'].keys():
                data.append(round(outputs['Channels'][c]['Corrected MTF50']*outputs['Size'][0 if outputs['Orientation'] == 'Horizontal' else 1]*2))
            writer.writerow(data)
            
            data = ['MTF50P Cy/pxl']
            for c in outputs['Channels'].keys():
                data.append(round(outputs['Channels'][c]['MTF50P'],4))
            writer.writerow(data)
            
            data = ['MTF50P LW/PH']
            for c in outputs['Channels'].keys():
                data.append(round(outputs['Channels'][c]['MTF50P']*outputs['Size'][0 if outputs['Orientation'] == 'Horizontal' else 1]*2))
            writer.writerow(data)
            
            data = ['Oversharpening %']
            for c in outputs['Channels'].keys():
                data.append(round(outputs['Channels'][c]['Sharpening']*100,2))
            writer.writerow(data)
            
            data = ['Sharpening Radius']
            for c in outputs['Channels'].keys():
                data.append(outputs['Channels'][c]['Sharpening Radius'])
            writer.writerow(data)
            writer.writerow('')
            
            # export MTF data
            header = ['Cy/Pxl','LW/PH']
            for c in outputs['Channels'].keys():
                header.append('MTF({0:s})'.format(c))
                header.append('MTF({0:s} corr)'.format(c))
            writer.writerow(header)
            for i in range(len(outputs['Cy/Pxl'])):
                data = [round(outputs['Cy/Pxl'][i],4), round(outputs['LW/PH'][i],2)]
                for c in outputs['Channels'].keys():
                    data.append(round(outputs['Channels'][c]['MTF'][i],4))
                    data.append(round(outputs['Channels'][c]['Corrected MTF'][i],4))
                writer.writerow(data)
            writer.writerow('')
            
            # export LSF data
            header = ['x (pixels)']
            for c in outputs['Channels'].keys():
                header.append('{0:s} Edge'.format(c))
            writer.writerow(header)
            data_length = len(outputs['Channels']['L']['LSF'])
            oversampling_rate = outputs['Oversampling']
            for i in range(data_length):
                data = [(i-data_length/2)/oversampling_rate]
                for c in outputs['Channels'].keys():
                    data.append(round(outputs['Channels'][c]['LSF'][i],2))
                writer.writerow(data)
            writer.writerow('')
            
            # export ESF data
            header = ['x (pixels)']
            for c in outputs['Channels'].keys():
                header.append('{0:s} Level'.format(c))
            writer.writerow(header)
            data_length = len(outputs['Channels']['L']['ESF'])
            for i in range(data_length):
                data = [(i-data_length/2)/oversampling_rate]
                for c in outputs['Channels'].keys():
                    data.append(round(outputs['Channels'][c]['ESF'][i],0))
                writer.writerow(data)
            writer.writerow('')