## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, count_fingers.py
## 3. In this challenge, you are only permitted to import numpy, and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
import sys
#It's ok to import whatever you want from the local util module if you would like:
sys.path.append('../util')
from data_handling import breakIntoGrids, reshapeIntoImage

def count_fingers(im):
    '''
    Example submission for coding challenge. 
    
    Args: im (nxm) unsigned 8-bit grayscale image 
    Returns: One of three integers: 1, 2, 3
    
    '''

    ## ------ Input Pipeline Develped in this Module ----- ##
    #You may use the finger pixel detection pipeline we developed in this module:
    #You may also replace this code with your own pipeline if you prefer
    im = im[0:30,0:35]
    im = im > 92 #Threshold image
#     print(im.shape)
#     X = breakIntoGrids(im, s = 9) #Break into 9x9 grids
    
    #Use rule we learned with decision tree
#     treeRule1 = lambda X: np.logical_and(np.logical_and(X[:, 40] == 1, X[:,0] == 0), X[:, 53] == 0)
#     yhat = treeRule1(X)
    
#     #Reshape prediction ino image:
#     yhat_reshaped = reshapeIntoImage(yhat, im.shape)

#     ## ----- Your Code Here ---- ##
    labels = [1, 2, 3]
# #     print(yhat_reshaped)
#     print("yhat",yhat,yhat_reshaped.shape)
    _,y_coords = np.where(im == 1)
    count = len(y_coords)
#     print(count)
    limit1 = 150 
    limit2 = 235
    print(limit1, limit2)
    if count >= 0 and count < limit1:
        return labels[0]
    elif count >= limit1 and count < limit2:
        return labels[1]
    else:
        return labels[2]
    
#     if count >= 0 and count < 100:
#         return labels[0]
#     elif count >= 100 and count < 150:
#         return labels[1]
#     else:
#         return labels[2]

    print("0 - 100 , 100 - 150 ")

    ## ----- Get rid of this! ---- ##
    #Let's guess randomly! Maybe we'll get lucky.
    
#     random_integer = np.random.randint(low = 0, high = 3)
#     return labels[random_integer]
    