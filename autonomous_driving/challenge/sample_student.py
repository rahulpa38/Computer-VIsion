## ---------------------------- ##
##
## Example student submission code for autonomous driving challenge.
## You must modify the train and predict methods and the NeuralNetwork class. 
## 
## ---------------------------- ##

import numpy as np
import cv2
import time
from scipy import signal as signal_sci, optimize
from numpy.ma import argmax

proportion_x = 0.04
proportion_y = 0.04

def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''

    #Hyper params
    data = np.genfromtxt(csv_file, delimiter=',')
    frame_nums = data[:, 0]
    steering_angles = data[:, 1]
    training_X = []


    
    for i in range(len(frame_nums)):
        frame_num = int(frame_nums[i])
        path = path_to_images + '/' + str(int(i)).zfill(4) + '.jpg'
        # read the image
        im = cv2.imread(path, 0)
        # resizing the image

        im = cv2.resize(im, (0, 0), fx= proportion_x, fy = proportion_y)
        # Normalizr the image
        im = im / 255        
        training_X.append(im.ravel())
        
    training_X = np.asarray(training_X)  
    
    training_Y = np.zeros([1500,64])
    array = [0.2, 0.33, 0.62, 0.90, 1]
    
    for i in range(len(steering_angles)):
        index = int(np.interp(steering_angles[i], [-171, 30], [0, 64]))
        if ((index + 4) <= 63) and ((index - 4) >= 0):
            training_Y[i][index - 4:index + 1] += array
            training_Y[i][index + 1:index + 5] = array[::-1][1:5]
#             print(training_Y[i][index - 4 :index + 5],array[::-1][1:5])

    NN = NeuralNetwork()
    
    Trainobj = NNtrainer(NN)    
    
    Trainobj.trainNN(training_X,training_Y)

    return NN


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    
    im = cv2.imread(image_file,0)
    im = cv2.resize(im, (0, 0), fx= proportion_x, fy = proportion_y)
    im= im.ravel()
    im = im / 255  
    
    
    ## 
    yHat = NN.forward(im)
    ## take the argumments
    argmx = argmax(yHat)
#     put it in interp
    angle = np.interp(argmx, [0, 64], [-171, 30])
    
    return angle


class NeuralNetwork(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2226
        self.outputLayerSize = 64
        self.hiddenLayerSize = 65

        # Weights (parameters)
        limit = np.sqrt(6 / (self.inputLayerSize + self.hiddenLayerSize))
        self.W1 = np.random.uniform(-limit, limit, (self.inputLayerSize, self.hiddenLayerSize))

        limit = np.sqrt(6 / (self.hiddenLayerSize + self.outputLayerSize))
        self.W2 = np.random.uniform(-limit, limit, (self.hiddenLayerSize, self.outputLayerSize))

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)       
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        # print('sigmoidprime shape',(np.exp(-z)/((1+np.exp(-z))**2)).shape)
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * np.sum((y - self.yHat) ** 2)
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        # print('delta3 shape: ',delta3.shape)
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    # Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
    
    
class NNtrainer(object):
    def __init__(self, NN):
        self.NN = NN
       
    def trainNN(self, training_X, training_Y):
        iterations = 3200
        alpha = 0.001
        bt1 = 0.9
        bt2 = 0.999
        epsl = 1e-08
        mvec1 = np.zeros(len(self.NN.getParams()))  # Initialize first moment vector
        mvec2 = np.zeros(len(self.NN.getParams()))  # Initialize second moment vector
        i = 0.0
        
        for itr in range(iterations):
            i += 1
            gradient = self.NN.computeGradients(X=training_X, y=training_Y)
            mvec2 = bt2*mvec2 + (1-bt2)*gradient**2
            mvec1 = bt1*mvec1 + (1-bt1)*gradient
            vt_hat = mvec2/(1-bt2**i)
            mt_hat = mvec1/(1-bt1**i)
            params = self.NN.getParams()
            parameters_new = params - alpha*mt_hat/(np.sqrt(vt_hat)+epsl)
            self.NN.setParams(parameters_new)
