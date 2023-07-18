#!/usr/bin/env python
# coding: utf-8

# In[17]:


import cv2
import numpy as np

# Read the image
img = cv2.imread('p1_image1.png')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the color ranges for white and red pieces in the HSV color space
white_range = np.array([[0, 0, 150], [255, 80, 255]])
red_range = np.array([[0.095, 100, 100], [10, 255, 255], [178, 100, 100], [179, 255, 255]])

# Threshold the image based on the color ranges
white_mask = cv2.inRange(hsv, white_range[0], white_range[1])
red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the masks and draw circles around them
contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 15 and radius < 50:
        cv2.circle(img, center, radius, (255, 0, 0), 3)

contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 15 and radius < 50:
        cv2.circle(img, center, radius, (255, 0, 0), 3)

# Save the output image
cv2.imwrite('output.png', img)


# In[1]:


import cv2
import numpy as np

# Read the image
img = cv2.imread('p1_image2.png')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the color ranges for white and red pieces in the HSV color space
white_range = np.array([[0, 0, 150], [255, 80, 255]])
red_range = np.array([[0.095, 100, 100], [10, 255, 255], [178, 100, 100], [179, 255, 255]])

# Threshold the image based on the color ranges
white_mask = cv2.inRange(hsv, white_range[0], white_range[1])
red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the masks and draw circles around them
contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 15 and radius < 50:
        cv2.circle(img, center, radius, (255, 0, 0), 3)

contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 15 and radius < 50:
        cv2.circle(img, center, radius, (255, 0, 0), 3)

# Save the output image
cv2.imwrite('output2.png', img)


# In[63]:


import cv2
import numpy as np

# Read the image
img = cv2.imread('p1_image3.png')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the color ranges for white and red pieces in the HSV color space
white_range = np.array([[0, 0, 155], [255, 255, 255]])
red_range = np.array([[0.095, 100, 100], [10, 255, 255], [178, 100, 100], [179, 255, 255]])

# Threshold the image based on the color ranges
white_mask = cv2.inRange(hsv, white_range[0], white_range[1])
red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the masks and draw circles around them
contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 15 and radius < 50:
        cv2.circle(img, center, radius, (255, 0, 0), 4)

contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 15 and radius < 50:
        cv2.circle(img, center, radius, (255, 0, 0), 4)

# Save the output image
cv2.imwrite('output3.png', img)

