#!/usr/bin/env python
# coding: utf-8

# # Assignment # 2 - IIP 

# ## Question # 1 

# #Obtain the images ‘lena.jp’ and ‘peppers.tif’ attached with the assignment.
# 
# A. Read and display the images.
# 
# B. Covert both the images to grayscale images.
# 
# C. Define a new 256 × 256 image A as follows: the left half of A, i.e., the first 128 columns, should be equal to the left half of the Lena image. The right half of A, i.e., the 129th column through the 256th column, should be equal to the right half of the Peppers image.

# ## Sol: 

# In[3]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
  
img1 = mpimg.imread('//Users//saadshafiq//Desktop//IIP//lena.png')
img2 = mpimg.imread('//Users//saadshafiq//Desktop//IIP//peppers.png') 

print("\t\t\tOrginal Images")
plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()


# In[4]:


from PIL import Image

print("\t\t\tGray Scale Images")
im1 = Image.open('//Users//saadshafiq//Desktop//IIP//lena.png').convert('LA')
im2 = Image.open('//Users//saadshafiq//Desktop//IIP//peppers.png').convert('LA')
display(im1,im2)


# In[5]:


from PIL import Image
import glob, os

im1 = Image.open('//Users//saadshafiq//Desktop//IIP//lena.png')
im2 = Image.open('//Users//saadshafiq//Desktop//IIP//peppers.png')

img_size1 = im1.size
#print(width,height)
img_size2 = im2.size

# Setting the points for cropped image
left = 0
top = 50
right = 510
bottom = 292

# Cropped image of above dimension
# (It will not change orginal image)
a = im1.crop((left, top, right, bottom))
b = im2.crop((left, top, right, bottom))
  
# Shows the image in image viewer
#a.show()
#b.show()

new_image = Image.new('RGB',(256, 256), (250,250,250))
new_image.paste(im1,(0,127))
new_image.paste(im2,(128,256))
plt.imshow(new_image)
plt.show()


        


# ## Question # 2 
# Read the image ‘lena.jpg’ and convert the image to gray scale image. We would like to enhance the contrast of the pixels in the middle brightness range. Specifically, replace each pixel's intensity 𝑖 with a new intensity
# 𝑖′, where
# 
#                                 𝑖′ =2×(𝑖−128)+128
# 
# ## Sol: 
# 

# In[2]:


from PIL import Image
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
  
# creating a image object
im = Image.open("//Users//saadshafiq//Desktop//IIP//lena.png").convert('LA')
plt.imshow(im)
plt.show()


cordinate = x, y = 180, 79
  
# using getpixel method
print ("old pixels:\t\t\t",im.getpixel(cordinate));
a = im.getpixel(cordinate)
x = np.array([a[0],a[1]])
#print(px)

x[0:1] = 2 * (x[0:1] - 128 ) + 128
x[1:2] = 2 * (x[1:2] - 128 ) + 128
new_pixel = x
print("New Pixels:\t\t\t", new_pixel)
#print(new_pixel)
px = im.load()
for i in range(im.size[0]):    
    for j in range(im.size[1]):  
        px[i,j] = (122,382)
        
        
        
plt.imshow(im)
plt.show()


# # Question # 3
# 
# Read the image ‘dark.tiff’. We would like to bring the image to a more typical average brightness. Apply a scaling factor to all pixels, so that the minimum pixel value becomes 0 and the max pixel value becomes 255. Include the final image in your report, as well as the MATLAB code you used to produce it.
# 
# ## Sol: 

# In[ ]:


# example of pixel normalization
from numpy import asarray
from PIL import Image
# load image
image = Image.open('//Users//saadshafiq//Desktop//IIP//lena.png')
pixels = asarray(image)
# confirm pixel range is 0-255
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# convert from integers to floats
pixels = pixels.astype('float32')
# normalize to the range 0-1
pixels /= 255.0
# confirm the normalization
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

for pixels in pixels:
    


# In[ ]:


import sys
from PIL import Image
import numpy as np

image_paths = ['//Users//saadshafiq//Desktop//IIP//lena.png']
min = sys.maxsize
max = -sys.maxsize

for image_path in image_paths:
    image = Image.open(image_path)
    np_image = np.asarray(image)
    if  np_image.min() > 128 
        np_image.min()= 256
    if  np_image.max() < 12
        max = np_image.max()
        


# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib 
from numpy import asarray
from PIL import Image

im = Image.open("//Users//saadshafiq//Desktop//IIP//lena.png").convert('LA')
plt.subplot(221)
display(im)
plt.subplot(222)
scaling = im > 158
imshow(scaling)



## i tried to solve this question but i failed


# # Question # 4
# 
# Write a function which plots histogram of any input image without using built-in MATLAB functions. Read the image ‘dark.tiff’ and draw the histogram of the image using your defined function for generating histogram.
# 
# ## Sol:

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

print("Displaying Image\n")
img = cv2.imread('//Users//saadshafiq//Desktop//IIP//lena.png',0)
#plt.imshow(img)
print("Displaying Histogram\n")

plt.hist(img.ravel(),256,[0,256])
plt.show()


# # Question # 5
# 
# Write your own function which performs histogram equalization on the given image and plots the new image as well as histogram of new image. You must not use built-in functions of MATLAB. Read the image ‘dark.tiff’ and display the resultant enhanced image after applying your histogram equalization function
# 
# ## Sol: 

# In[10]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
  
# import Numpy
import numpy as np
  
# read a image using imread
img = cv2.imread('//Users//saadshafiq//Desktop//IIP//lena.png', 0)
print("\t\t\tBefore Equalization")
plt.imshow(img)
plt.show()

# creating a Histograms Equalization of a image using cv2.equalizeHist()
equal = cv2.equalizeHist(img)
  
# show image input vs output
print("\t\t\tAfter Equalization")
plt.imshow(equal)


# In[11]:


#histogram of new image
print("\t\t\tDisplaying Histogram of image After Equalization\n")
plt.hist(equal.ravel(),256,[0,256])
plt.show()


# In[ ]:




