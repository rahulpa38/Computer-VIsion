## ---------------------------- ##
##
## Example student submission code for autonomous driving challenge.
## You must modify the train and predict methods and the NeuralNetwork class. 
## 
## ---------------------------- ##

import numpy as np
import cv2
from tqdm import tqdm
import time
import os


image_size = 32
cropped_image = int(image_size/2)
# output_layer_size = 1
# Let 's go from one output layer size to multiple output layer as it was not enough to train
output_layer_size = 1
min_steering_angle = 0
# bins = []
max_steering_angle = 0

def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''
    global min_steering_ang ,max_steering_angle
    resized_images = []

    # You may make changes here if you wish. 
    # Import Steering Angles CSV
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    steering_angles = np.reshape(steering_angles,(-1,1))
    print(steering_angles.shape)
# #     steering_angles_1 = np.matrix(angles).transpose()
#     max_steering_angle = int(np.max(steering_angles_1))
#     min_steering_angle = int(np.min(steering_angles_1))
    
    ## Bins Creation
#     bins = np.linspace(min_steering_angle,max_steering_angle,output_layer_size) 
#     numeric_values = np.linspace(0,output_layer_size -1,output_layer_size)
#     steering_angles = np.zeros((len(steering_angles_1),output_layer_size))
#     for i,angle in enumerate(steering_angles_1):
#         current_index =  int(np.interp(angle,bins, numeric_values))
#         steering_angles[i,current_index] = 1
# # not ignoring the neighbouring steering angles since they could have also been the actual steering angles
#         if current_index - 1 >= 0 and current_index + 1 < output_layer_size:
#             steering_angles[i,current_index - 1] = 0.89
#             steering_angles[i,current_index + 1] = 0.89
#             if current_index - 2 >= 0 and  current_index + 2 < output_layer_size:
#                 steering_angles[i,current_index - 2] = 0.6
#                 steering_angles[i,current_index + 2] = 0.6
#                 if current_index - 3 >= 0 and  current_index + 3 < output_layer_size:
#                     steering_angles[i,current_index - 2] = 0.3
#                     steering_angles[i,current_index + 2] = 0.3
#                     if current_index - 4 >= 0 and  current_index + 4 < output_layer_size:
#                         steering_angles[i,current_index - 2] = 0.1
#                         steering_angles[i,current_index + 2] = 0.1
        
## You could import your images one at a time or all at once first, 
# here's some code to import a single image:
    for i,frame in enumerate(frame_nums):
        ## Read  the image
        im_full = cv2.imread('training' + '/resized_images_32/' + str(int(frame)).zfill(4) + '.jpg',cv2.IMREAD_UNCHANGED)
        ## Gray Color
#         im_full = cv2.cvtColor(im_full, cv2.COLOR_RGB2GRAY)
#         im_full = cv2.resize(im_full, (image_size,image_size)) 
        ## Image Vector
        image_vector = np.array(im_full[cropped_image:,:])
        ## Normalize the image
        image_vector = image_vector/255
        
        resized_images.append(image_vector)
        
        
    # Train your network here. You'll probably need some weights and gradients!
   
    resized_images = np.reshape(resized_images,(1500,layer1))
    
    for iter in range(iterations):
        
        ## Compute Gradients DJ/DW1 DJ/DW2
        grads  = NN.computeGradients(resized_images, steering_angles)
#          get W1 and W2
        params = NN.getParams()   
        
#         Update Weights
        params[:] = params[:] - ((learning_rate*grads[:])/(len(resized_images)))                                       
        ## update weights in Class Neural Network
        ## Set Weights
        NN.setParams(params)
        
        ## Print Cost Function
        if iter%100 == 0:
            print("------iter--------- ",iter)
            cost_function = NN.costFunction(resized_images[int(iter/100)],steering_angles[int(iter/100)])
            print ("Cost Function",cost_function )
#             if cost_function < 0.015 and iter > 3500:
#                 break

    return NN



def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    bins = np.linspace(min_steering_angle,max_steering_angle,64)
    
    ## Transform Image and Normalize pixels
    im_full = cv2.imread(image_file)    
    im_full = cv2.cvtColor(im_full, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(im_full, (image_size,image_size)) 
    
    ## Perform inference using your Neural Network (NN) here.
    image_vector = np.array(resized_image[cropped_image:,:])
    image_vector = np.reshape(image_vector,(1,-1))
    ## Normalize the image
    image_vector = image_vector/255
    
    yhat = NN.forward(image_vector)
#     yhat = bins[np.argmax(yhat)]
    
    return yhat

class NeuralNetwork(object):
    def __init__(self,input_layer_size,hidden_layer_size,output_layer_size):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = input_layer_size
        self.outputLayerSize = output_layer_size
        self.hiddenLayerSize = hidden_layer_size
        
        ## Initialize weights properly 
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)*np.sqrt(1/(self.inputLayerSize + self.hiddenLayerSize))
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)*np.sqrt(1/(self.outputLayerSize + self.hiddenLayerSize))
    
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.multiply(np.dot(delta3, self.W2.T),self.sigmoidPrime(self.z2))
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
        
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
        
        
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))