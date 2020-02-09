#!/usr/bin/env python
# coding: utf-8

# In[103]:


import numpy as np
import os, sys
sys.path.append('..')
get_ipython().run_line_magic('pylab', 'inline')
from util.filters import filter_2d
from util.image import convert_to_grayscale

# im = imread('../data/easy/ball/ball_2.jpg')
gray = convert_to_grayscale(im/255.)
plt.imshow(gray, cmap = 'gray')
Kx = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])

Ky = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])

Gx = filter_2d(gray, Kx)
Gy = filter_2d(gray, Ky)

#Compute Gradient Magnitude and Direction:
G_magnitude = np.sqrt(Gx**2+Gy**2)
G_direction = np.arctan2(Gy, Gx)
fig = figure(0, (6,6))
plt.imshow(G_magnitude)


# In[104]:


from ipywidgets import interact

#Show all pixels with values above threshold:
def tune_thresh(thresh = 0):
    fig = figure(0, (8,8))
    imshow(G_magnitude > thresh)


# In[105]:


interact(tune_thresh, thresh = (0, 2.0, 0.05))


# In[106]:


edges = G_magnitude > 1.05


# In[107]:


fig = figure(0, (6,6))
imshow(edges)


# In[108]:


y_coords, x_coords = np.where(edges)


# In[109]:


fig = figure(0, (20,8))
fig.add_subplot(1,2,1)
imshow(edges)

fig.add_subplot(1,2,2)
scatter(x_coords, y_coords, s = 5)
grid(1)


# In[110]:


y_coords_flipped = edges.shape[0] - y_coords


# In[111]:


fig = figure(0, (16,8))
ax = fig.add_subplot(1,2,1)
imshow(edges)

ax2 = fig.add_subplot(1,2,2)
scatter(x_coords, y_coords_flipped, s = 5)
grid(1)
xlim([0, edges.shape[0]]);
ylim([0, edges.shape[0]]);


# In[112]:


#How many bins for each variable in parameter space?
phi_bins = 128
theta_bins = 128

accumulator = np.zeros((phi_bins, theta_bins))


# In[113]:


rho_min = -edges.shape[0]*np.sqrt(2)
rho_max = edges.shape[1]*np.sqrt(2)

theta_min = 0
theta_max = np.pi

#Compute the rho and theta values for the grids in our accumulator:
rhos = np.linspace(rho_min, rho_max, accumulator.shape[0])
thetas = np.linspace(theta_min, theta_max, accumulator.shape[1])


# In[114]:


for i in range(len(x_coords)):
    #Grab a single point
    x = x_coords[i]
    y = y_coords_flipped[i]

    #Actually do transform!
    curve_rhos = x*np.cos(thetas)+y*np.sin(thetas)

    for j in range(len(thetas)):
        #Make sure that the part of the curve falls within our accumulator
        if np.min(abs(curve_rhos[j]-rhos)) <= 1.0:
            #Find the cell our curve goes through:
            rho_index = argmin(abs(curve_rhos[j]-rhos))
            accumulator[rho_index, j] += 1


# In[115]:


fig = figure(0, (8,8))
imshow(accumulator);


# In[116]:


from mpl_toolkits.mplot3d import Axes3D

fig = figure(figsize=(16, 16));
ax1 = fig.add_subplot(111, projection='3d')

_x = np.arange(accumulator.shape[0])
_y = np.arange(accumulator.shape[1])
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = accumulator.ravel()
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade = True);


# In[117]:


max_value = np.max(accumulator)


# In[118]:


max_value


# In[119]:


def tune_thresh(relative_thresh = 0.9):
    fig = figure(0, (8,8))
    imshow(accumulator > relative_thresh * max_value)


# In[120]:


interact(tune_thresh, relative_thresh = (0, 1, 0.05))


# In[121]:


relative_thresh = 0.6

#Indices of maximum theta and rho values
rho_max_indices, theta_max_indices,  = np.where(accumulator > relative_thresh * max_value)


# In[122]:


theta_max_indices, rho_max_indices


# In[123]:


thetas_max = thetas[theta_max_indices]
rhos_max = rhos[rho_max_indices]


# In[124]:


fig = figure(0, (8,8))
imshow(im)
 
for theta, rho in zip(thetas_max, rhos_max):
    #x-values to use in plotting:
    xs = np.arange(im.shape[1])
    
    #Check if theta == 0, this would be a vertical line
    if theta != 0:
        ys = -cos(theta)/sin(theta)*xs + rho/sin(theta)
        
    #Special handling for plotting vertical line:
    else:
        xs = rho*np.ones(len(xs))
        ys = np.arange(im.shape[0])
    
    #have to re-flip y-values to reverse the flip we applied initially:
    plot(xs, im.shape[0]-ys)
    
xlim([0, im.shape[0]]);
ylim([im.shape[1], 0]);


# In[ ]:





# In[ ]:




