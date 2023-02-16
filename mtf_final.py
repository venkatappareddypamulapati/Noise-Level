import numpy as np
from argparse import ArgumentParser
import os.path
from scipy import signal
import specutils
from specutils.analysis import gaussian_sigma_width, gaussian_fwhm, fwhm, fwzi
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
from scipy import interpolate
from scipy.signal import savgol_filter
import os, sys
from csv import writer
import pandas as pd
import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage.util import random_noise
from scipy import signal
#from medpy.filter.smoothing import anisotropic_diffusion
import cv2
from csv import writer
import random

# Reference:
# http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs."""


    return np.isnan(y), lambda z: z.nonzero()[0]

class EventHandler(object):   #class, part of events, which simply is responsible for managing all callbacks (like clicks) which are to be executed when invoked.

    def __init__(self, filename): #initialize variables
        self.filename = filename

    def line_select_callback(self, eclick, erelease):
        #Callback for line selection.
        #eclick and erelease are the press and release events
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print('clicked : ',x1, y1, x2, y2)
        self.roi = np.array([y1, y2, x1, x2])
        print('shape of roi : ',self.roi.shape)
    def event_exit_manager(self, event):
        if event.key in ['enter']:
            PDS_Compute_MTF(self.filename, self.roi)

class ROI_selection(object):
    # selecting ROI from an image
    def __init__(self, filename):
        self.filename = filename
        self.image_data = cv2.imread(filename, 0)
        fig_image, current_ax = plt.subplots()
        plt.imshow(self.image_data, cmap='gray')
        eh = EventHandler(self.filename)
        rectangle_selector = RectangleSelector(current_ax,
                                               eh.line_select_callback,
                                               useblit=True,
                                               button=[1, 2, 3],
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        plt.connect('key_press_event', eh.event_exit_manager)
        plt.show()


class PDS_Compute_MTF(object): # calculation MTF

    def __init__(self, filename, roi):
        image_data = cv2.imread(filename, 0)
        roi = roi.astype(int)
        image_data = image_data[91:210, 113:148]
        # select the patch of an image
        self.data = image_data

        print('Data Shape Cropped ROI',self.data.shape)

        # Threshold the image
        #_, th = cv2.threshold(self.data,0, 255, cv2.THRESH_OTSU+cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
       # _, th = cv2.threshold(self.data, 0, 255, cv2.THRESH_OTSU)
       # _, th = cv2.threshold(self.data, 0, 255, cv2.THRESH_OTSU + cv2.ADAPTIVE_THRESH_MEAN_C)
        _, th = cv2.threshold(self.data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        self.min = np.amin(self.data)
        self.max = np.amax(self.data)
        self.threshold = th * (self.max - self.min) + self.min

        below_thresh = ((self.data >= self.min) & (self.data <= self.threshold))
        above_thresh = ((self.data >= self.threshold) & (self.data <= self.max))
        area_below_thresh = self.data[below_thresh].sum() / below_thresh.sum()
        area_above_thresh = self.data[above_thresh].sum() / above_thresh.sum()
        self.threshold = (area_below_thresh - area_above_thresh) / 2 + area_above_thresh
       # Detects the edges
        edges = cv2.Canny(self.data, self.min, self.max)
        fig = plt.figure()
        row_edge, col_edge = np.where(edges == 255)
        print(row_edge)
        print(col_edge)
        z = np.polyfit(np.flipud(col_edge),row_edge,1)
        angle_radians = np.arctan(z[0])
        angle_deg = angle_radians * (180 / np.pi)
        print('Angle',angle_deg)
        if  abs(angle_deg)<45 :
            self.data = np.transpose(self.data)
            print('Transposing!!')
        plt.subplot(2, 3, 1)
        plt.imshow(self.data, cmap='gray')
        plt.title("Image")
        plt.subplot(2, 3, 2)
        plt.imshow(edges, cmap='gray')
        plt.title("Detected Edge")
        plt.subplot(2, 3, 3)
        plt.plot(np.flipud(col_edge), row_edge)
        plt.title('Col_edge vs Row_edge')
        plt.ylim(100, 0)
        self.compute_esf()
    # edge spread function calculation
    def compute_esf(self):
        smooth_img=self.data
        row = self.data.shape[0]
        column = self.data.shape[1]
        array_values_near_edge = np.empty([row, 13])
        array_positions = np.empty([row, 13])
        edge_pos = np.empty(row)
        smooth_img = smooth_img.astype(float)
        for i in range(0, row):
            diff_img = smooth_img[i, 1:] - smooth_img[i, 0:(column - 1)]
            abs_diff_img = np.absolute(diff_img)
            abs_diff_max = np.amax(abs_diff_img)
            if abs_diff_max < 1:
                raise IOError('No Edge Found')
            app_edge = np.where(abs_diff_img == abs_diff_max)
            bound_edge_left = app_edge[0][0] - 2
            bound_edge_right = app_edge[0][0] + 3
            strip_cropped = self.data[i, bound_edge_left:bound_edge_right]
            temp_y = np.arange(1, 6)
            #f = interpolate.interp1d(strip_cropped, temp_y, kind='cubic')
            f=interpolate.interp1d(strip_cropped, temp_y, kind='cubic', axis=- 1, copy=True, bounds_error=False, fill_value="extrapolate", assume_sorted=False)
            edge_pos_temp = f(self.threshold)
            edge_pos[i] = edge_pos_temp + bound_edge_left - 1
            bound_edge_left_expand = app_edge[0][0] - 6
            bound_edge_right_expand = app_edge[0][0] + 7
            array_values_near_edge[i, :] = self.data[i, bound_edge_left_expand:bound_edge_right_expand]
            array_positions[i, :] = np.arange(bound_edge_left_expand, bound_edge_right_expand)
        y = np.arange(0, row)
        nans, x = nan_helper(edge_pos)
        edge_pos[nans] = np.interp(x(nans), x(~nans), edge_pos[~nans])

        array_positions_by_edge = array_positions - np.transpose(edge_pos * np.ones((13, 1)))
        num_row = array_positions_by_edge.shape[0]
        num_col = array_positions_by_edge.shape[1]
        array_values_by_edge = np.reshape(array_values_near_edge, num_row * num_col, order='F')
        array_positions_by_edge = np.reshape(array_positions_by_edge, num_row * num_col, order='F')

        bin_pad = 0.0001
        pixel_subdiv = 0.10
        topedge = np.amax(array_positions_by_edge) + bin_pad + pixel_subdiv
        botedge = np.amin(array_positions_by_edge) - bin_pad
        binedges = np.arange(botedge, topedge + 1, pixel_subdiv)
        numbins = np.shape(binedges)[0] - 1

        binpositions = binedges[0:numbins] + (0.5) * pixel_subdiv

        h, whichbin = np.histogram(array_positions_by_edge, binedges)
        whichbin = np.digitize(array_positions_by_edge, binedges)
        binmean = np.empty(numbins)

        for i in range(0, numbins):
            flagbinmembers = (whichbin == i)
            binmembers = array_values_by_edge[flagbinmembers]
            binmean[i] = np.mean(binmembers)
        nans, x = nan_helper(binmean)
        binmean[nans] = np.interp(x(nans), x(~nans), binmean[~nans])
        esf = binmean
        xesf = binpositions
        xesf = xesf - np.amin(xesf)
        self.xesf = xesf
        esf_smooth = savgol_filter(esf, 51, 3)
        self.esf = esf
        self.esf_smooth = esf_smooth
        plt.subplot(2, 3, 4)
        plt.title("ESF Curve")
        plt.xlabel("pixel")
        plt.ylabel("DN Value")
        plt.plot(xesf, esf_smooth)
        plt.grid()
        self.compute_lsf()
    # line spread function calculation
    def compute_lsf(self):
        diff_esf_smooth = abs(self.esf_smooth[0:(self.esf_smooth.shape[0] - 1)] - self.esf_smooth[1:])
        diff_esf_smooth = np.append(0, diff_esf_smooth)
        lsf_smooth = diff_esf_smooth
        self.lsf_smooth = lsf_smooth
        #win = signal.windows.hamming(51)
        #lsf_smooth = signal.convolve(self.lsf_smooth, win, mode='same') / sum(win)
        FWHM=np.max(lsf_smooth)/2
        print(lsf_smooth)
        plt.subplot(2, 3, 5)
        plt.title("LSF Curve")
        plt.xlabel("pixel")
        plt.ylabel("DN Value")
        plt.plot(self.xesf, lsf_smooth)
        st1=self.xesf[np.argmin(np.abs(lsf_smooth[0:lsf_smooth.shape[0]//2]-FWHM))]
        plt.plot(st1,FWHM,'g*')
        st2=self.xesf[np.argmin(np.abs(lsf_smooth[lsf_smooth.shape[0]//2:]-FWHM))+lsf_smooth.shape[0]//2]
        plt.plot(st2,FWHM,'m*')
        plt.text((st1+st2)/2, FWHM+0.1,'FWHM=%3.1f' %(st2-st1), ha="center", va="center")
        plt.plot([st1, st2], [FWHM,FWHM], color='r', linestyle='--')
        plt.axvline(st1, color='green', lw=2, alpha=0.5)
        plt.axvline(st2, color='green', lw=2, alpha=0.5)
        plt.grid(1)
        self.compute_mtf()
    # MTF computation
    def compute_mtf(self):
        mtf_smooth = np.absolute(np.fft.fft(self.lsf_smooth, 2048))
        mtf_final_smooth = np.fft.fftshift(mtf_smooth)
        plt.subplot(2, 3, 6)
        x_mtf_final = np.arange(0, 1, 1. / 127)
        mtf_final_smooth = mtf_final_smooth[1024:1151] / np.amax(mtf_final_smooth[1024:1151])
        mtf_required=mtf_final_smooth[31]
        mtf_result=[mtf_required]
        mtf.append(mtf_result)
        np.savetxt('/home/venkat/test1.csv', mtf, '%10.5f',
                   header='mtf',
                   delimiter=',')
        print(mtf_required)
        plt.plot(x_mtf_final, mtf_final_smooth)
        plt.xlabel("cycles/pixel")
        plt.xlim(0, 0.5)
        plt.ylabel("Modulation Factor")
        plt.title("MTF Curve")
        plt.grid()
        plt.show()
        return mtf_required

    # main function
if __name__ == '__main__':
    all_results=[]
    mtf=[]
    for size_of_kernel in range (5,7,2):
        for variance in np.arange(1, 3, .2):
             img = cv2.imread('/home/venkat/Downloads/img.tif') # read the image
             blur = cv2.GaussianBlur(img, (size_of_kernel, size_of_kernel), variance) # blur the image
             cv2.imwrite('/home/venkat/Downloads/blurred_img.tif', blur)
             filename="/home/venkat/Downloads/blurred_img.tif"
             roi = np.float32([43,179,98,155]) # fix the ROI initially
             PDS_Compute_MTF(filename, roi)
             result = [size_of_kernel, variance]
             all_results.append(result)
all_results = np.float32(all_results)
#result=np.zeros((all_results.shape[0],3))
np.savetxt('/home/venkat/test.csv', all_results, '%10.5f',
                   header='size_of_kernel, variance',
                   delimiter=',')
df_homes = pd.read_csv("/home/venkat/test.csv")
df_homes1 = pd.read_csv("/home/venkat/test1.csv")
# This method combines a list of pandas dataframes into one dataframe (MTF vs Kernels with different variances)
pd.concat([df_homes, df_homes1], axis=1).to_csv('/home/venkat/homes_complete_13_1.csv')