## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, classify.py
## 3. In this challenge, you are only permitted to import numpy and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
import sys
#sys.path.append('..')
#from util.image import convert_to_grayscale
# from util.filters import filter_2d 

#im = im[]

def convert_to_grayscale(im):
    '''
    Convert color image to grayscale.
    Args: im = (nxmx3) floating point color image scaled between 0 and 1
    Returns: (nxm) floating point grayscale image scaled between 0 and 1
    '''
    return np.mean(im, axis = 2)

#Show all pixels with values above threshold:
def tune_thresh(thresh = 0):
    fig = figure(0, (8,8))
    imshow(G_magnitude > thresh)

def filter_2d(im, kernel):
    '''
    Filter an image by taking the dot product of each 
    image neighborhood with the kernel matrix.
    Args:
    im = (H x W) grayscale floating point image
    kernel = (M x N) matrix, smaller than im
    Returns: 
    (H-M+1 x W-N+1) filtered image.
    '''

    M = kernel.shape[0] 
    N = kernel.shape[1]
    H = im.shape[0]
    W = im.shape[1]
    
    filtered_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')
    
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            image_patch = im[i:i+M, j:j+N]
            filtered_image[i, j] = np.sum(np.multiply(image_patch, kernel))
            
    return filtered_image
    
def classify(im):
    '''
    Example submission for coding challenge.

    Args: im (nxmx3) unsigned 8-bit color image
    Returns: One of three strings: 'brick', 'ball', or 'cylinder'

    '''
    im = im[10:245, 10:245]
    im_shape = im.shape
    gray = convert_to_grayscale(im/255.)
    Kx = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])

    Ky = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])

    Gx = filter_2d(gray, Kx)
    Gy = filter_2d(gray, Ky)

    G_magnitude = np.sqrt(Gx**2 + Gy**2)
    
    labels = ['ball','cylinder','brick']
    
    hist_values = np.histogram(G_magnitude)
    
    sum_hist_values = 0
    
    for i in range(int((len(hist_values))/2) + 1,len(hist_values[0])):
        sum_hist_values += hist_values[0][i]*hist_values[1][i]
    if sum_hist_values > 0 and sum_hist_values <=1500:
        return labels[0]
    elif sum_hist_values > 1500 and sum_hist_values <= 1750:
        return labels[1]
    else:
        return labels[2]