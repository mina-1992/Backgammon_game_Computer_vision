#!/usr/bin/env python
# coding: utf-8

# In[137]:


import cv2
import numpy as np

# Read the image
img = cv2.imread('p2_image1.png')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the color range for black in the HSV color space
black_range = np.array([[0, 0, 0], [255, 255, 50]])

# Define the color range for white in the HSV color space
white_range = np.array([[0, 0, 200], [255, 30, 255]])

# Threshold the image based on the color ranges
black_mask = cv2.inRange(hsv, black_range[0], black_range[1])
white_mask = cv2.inRange(hsv, white_range[0], white_range[1])

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)
black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the black mask and draw circles around them
contours_black, hierarchy = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
centroids_black = []
for cnt in contours_black:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 10 and radius < 14:
        # Draw a circle around the black circle
        #cv2.circle(img, center, radius, (0, 0, 255), 2)

        # Save the centroid of the black circle
        centroids_black.append(center)

# Find contours in the white mask and draw circles around them
contours_white, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
centroids_white = []
radii_white = []
for cnt in contours_white:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 60 and radius < 100:
        # Draw a circle around the white circle
        #cv2.circle(img, center, radius, (0, 255, 0), 2)

        # Save the centroid and radius of the white circle
        centroids_white.append(center)
        radii_white.append(radius)

# Count the number of black circles within a certain distance of each white circle
for i in range(len(centroids_white)):
    num_black_circles = 0
    for j in range(len(centroids_black)):
        dist = np.sqrt((centroids_white[i][0]-centroids_black[j][0])**2 + (centroids_white[i][1]-centroids_black[j][1])**2)
        if dist <= radii_white[i]/2:
            num_black_circles += 1

    # Write the sum of black circles above each dice
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(num_black_circles), (centroids_white[i][0], centroids_white[i][1]-int(radii_white[i])-20), font, 1, (255, 0, 0), 2, cv2.LINE_AA)


cv2.imwrite('opencv_output_image1.png', img)


# In[139]:


import cv2
import numpy as np

# Read the image
img = cv2.imread('p3_image3.png')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the color range for black in the HSV color space
black_range = np.array([[0, 0, 0], [255, 255, 50]])

# Define the color range for white in the HSV color space
white_range = np.array([[0, 0, 200], [255, 30, 255]])

# Threshold the image based on the color ranges
black_mask = cv2.inRange(hsv, black_range[0], black_range[1])
white_mask = cv2.inRange(hsv, white_range[0], white_range[1])

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)
black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the black mask and draw circles around them
contours_black, hierarchy = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
centroids_black = []
for cnt in contours_black:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 10 and radius < 14:
        # Draw a circle around the black circle
        #cv2.circle(img, center, radius, (0, 0, 255), 2)

        # Save the centroid of the black circle
        centroids_black.append(center)

# Find contours in the white mask and draw circles around them
contours_white, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
centroids_white = []
radii_white = []
for cnt in contours_white:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 60 and radius < 100:
        # Draw a circle around the white circle
        #cv2.circle(img, center, radius, (0, 255, 0), 2)

        # Save the centroid and radius of the white circle
        centroids_white.append(center)
        radii_white.append(radius)

# Count the number of black circles within a certain distance of each white circle
for i in range(len(centroids_white)):
    num_black_circles = 0
    for j in range(len(centroids_black)):
        dist = np.sqrt((centroids_white[i][0]-centroids_black[j][0])**2 + (centroids_white[i][1]-centroids_black[j][1])**2)
        if dist <= radii_white[i]/2:
            num_black_circles += 1

    # Write the sum of black circles above each dice
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(num_black_circles), (centroids_white[i][0], centroids_white[i][1]-int(radii_white[i])-20), font, 1, (255, 0, 0), 2, cv2.LINE_AA)


cv2.imwrite('opencv_output_image3.png', img)

