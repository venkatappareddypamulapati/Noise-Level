#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:05:07 2023

@author: venkat
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:39:39 2023

@author: venkat
"""
# Hello Venkat
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
# import tifffile as tiff
from skimage.util import random_noise
from numpy import random
import cv2
import os
import glob
from scipy import stats
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

folder_path = '/home/venkat/Downloads/noisy'

test = os.listdir(folder_path)

for images in test:
    if images.endswith(".png"):
        os.remove(os.path.join(folder_path, images))


def printcheckboard(n):
    """print("Checkerboard pattern:")"""

    # create a n * n matrix
    x = np.zeros((n, n), dtype=int)
    # fill with 1 the alternate rows and columns
    x[0:100, 0:100] = 5
    x[0:100, 100:200] = 10
    x[0:100, 200:300] = 20
    x[0:100, 300:400] = 30
    x[0:100, 400:500] = 25
    x[100:200, 100:200] = 100
    x[100:200, 200:300] = 70
    x[100:200, 300:400] = 120
    x[100:200, 0:100] = 100
    x[100:200, 400:500] = 90
    x[200:300, 0:100] = 100
    x[200:300, 100:200] = 85
    x[200:300, 200:300] = 111
    x[200:400, 300:400] = 118
    x[200:500, 400:500] = 95
    x[300:400, 0:100] = 156
    x[300:400, 100:200] = 168
    x[300:400, 200:300] = 220
    x[300:400, 300:400] = 135
    x[300:400, 400:500] = 75
    x[400:500, 0:100] = 156
    x[400:500, 100:200] = 185
    x[400:500, 200:300] = 130
    x[400:500, 300:400] = 55

    # print the pattern
    """for i in range(n):
        for j in range(n):
            print(x[i][j], end =" ")
        print()"""

    return x


# driver code
n = 500


# img=printcheckboard(n)

def poisson_noise(image, scale):
    noisy_image = np.random.poisson(image * scale) / scale
    return noisy_image


img = cv2.imread('/home/venkat/data_set1/img6.png', 0)

for iter in range(1, 2):

    # noise=np.random.normal(0,1,img.shape)
    # print("Noise=", noise)
    # noise1=random.poisson(lam=0.1, size=img.shape)
    # img1=img+noise
    img1 = poisson_noise(img, .1)
    cv2.imwrite('/home/venkat/image.png', img1)
    # noise_std=np.std(noise)
    # img1=poisson_noise(img, .1)
    # img1=img1+noise
    # img=img+np.random.normal(0,1,img.shape)
    # print(img)
    # print(width)
    # print(height)
    patch_size = 11
    # step_size=int(input('enter the step_size'))
    patches_img = patchify(img1, (patch_size, patch_size), step=int(patch_size))
    # pixel_values=[i for i in range(np.amax(img))]
    counts = []
    std = []
    for x_cord in range(patches_img.shape[0]):
        for y_cord in range(patches_img.shape[1]):
            single_patch_img = patches_img[x_cord, y_cord, :, :]
            width, height = single_patch_img.shape
            pixel_value = single_patch_img[width // 2, height // 2]
            number = 0
            for width_pixel in range(width):
                for height_pixel in range(height):
                    if np.abs(single_patch_img[width // 2, height // 2] - single_patch_img[
                        width_pixel, height_pixel]) < 3 * np.std(single_patch_img):  # Gaussian Noise
                        # if np.abs(single_patch_img[width // 2, height // 2] - single_patch_img[ width_pixel, height_pixel]) < 0:
                        number += 1
                        # print(single_patch_img[width//2, height//2])
            if number <= width * height - 1:
                cv2.imwrite('/home/venkat/Downloads/noisy/' + 'image_' + '_' + str(x_cord) + str(y_cord) + ".png",
                            single_patch_img)
                counts_value = np.int32(np.average(single_patch_img[:]))
                std_value = (np.std(single_patch_img))
                counts.append(counts_value)
                std.append(std_value)
# print(counts)
# print(std)
# create a dictionary
test_dict = {k: v for k, v in zip(std, counts)}
# test_dict={4:1, 2:1, 3:2, 8:4, 5:4, 6:7}
"""temp=[]
res = dict()
for key, val in test_dict.items():
    if val not in temp:
        temp.append(val)
        res[key] = val

        #print(key)"""

# test_dict = {4:1, 2:1,3:1, 3:2, 8:4, 5:4,4:4, 6:7, 2:2}

# find the maximum value for each unique value in the dictionary
max_values = {}
for k, v in test_dict.items():
    if v not in max_values or max_values[v] > k:
        max_values[v] = k

# remove all key-value pairs where the value is less than or equal to the value of any other key
result = {k: v for k, v in test_dict.items() if k == max_values[v]}

# print(result)

# Print the resulting dictionary
# Printing resultant dictionary
# print("Resultant dictionary is : " + str(res))
value_counts = (list(result.values()))
# print(counts)
noise_std = (list(result.keys()))
# print(std)
# plt.plot(counts, std, '*')
# plt.xlabel('mean_pixel_value')
# plt.ylabel('standard_deviation_value')


# Calculate the linear regression line using scipy
slope, intercept, r_value, p_value, std_err = linregress(value_counts, noise_std)

# Print the slope and intercept
print("Slope:", slope)
print("Intercept:", intercept)

# Create the scatter plot
# plt.plot(counts, std, '*')
plt.scatter(value_counts, noise_std)

# Add the linear regression line to the plot
plt.plot(value_counts, intercept + slope * np.array(value_counts), 'b')

poly_fit = np.poly1d(np.polyfit(value_counts, noise_std, 1))

myline = sorted(value_counts)

plt.plot(myline, poly_fit(myline), c='r', linestyle='-')

# Set the title and axis labels
plt.title('Linear Regression Plot')

# Display the plot
plt.show()
"""
plt.figure()
plt.scatter(counts,std)

plt.show()"""



