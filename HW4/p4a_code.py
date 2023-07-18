#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# Read the input video file
cap = cv2.VideoCapture('p4_video2.m4v')

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video2a.mp4', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

# Define the color ranges for white pieces in the HSV color space
white_range = np.array([[0, 0, 157], [255, 150, 255]])
red_range = np.array([[0, 100, 100], [5, 255, 255], [179, 100, 100], [179, 255, 255]])

# Define the size range for white pieces
min_size = 2100
max_size = 5000

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)

# Process each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold the frame based on the color range
        white_mask = cv2.inRange(hsv, white_range[0], white_range[1])
        red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])
        
        # Apply morphological operations to the mask
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)    
        # Find contours of white pieces
        contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw filled circles around white pieces of certain size
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size < area < max_size:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 0), -1)


        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw filled circles around white pieces of certain size
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size < area < max_size:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (255, 255, 0), -1)
            #cv2.circle(frame, center, radius, (0, 255, 0), 4)  # Draw green circles for red pieces    
        # Write the processed frame to output video
        out.write(frame)

        # Display the processed frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release
cap.release()
out.release()


# In[2]:


import cv2
import numpy as np

# Read the input video file
cap = cv2.VideoCapture('p4_video1.m4v')

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video1a.mp4', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

# Define the color ranges for white pieces in the HSV color space
white_range = np.array([[0, 0, 157], [255, 150, 255]])
red_range = np.array([[0, 100, 100], [5, 255, 255], [179, 100, 100], [179, 255, 255]])

# Define the size range for white pieces
min_size = 2100
max_size = 5000

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)

# Process each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold the frame based on the color range
        white_mask = cv2.inRange(hsv, white_range[0], white_range[1])
        red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])
        
        # Apply morphological operations to the mask
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)    
        # Find contours of white pieces
        contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw filled circles around white pieces of certain size
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size < area < max_size:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 0), -1)


        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw filled circles around white pieces of certain size
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size < area < max_size:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (255, 255, 0), -1)
            #cv2.circle(frame, center, radius, (0, 255, 0), 4)  # Draw green circles for red pieces    
        # Write the processed frame to output video
        out.write(frame)

        # Display the processed frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release
cap.release()
out.release()


# In[3]:


import cv2
import numpy as np

# Read the input video file
cap = cv2.VideoCapture('p4_video3.m4v')

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video3a.mp4', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

# Define the color ranges for white pieces in the HSV color space
white_range = np.array([[0, 0, 155], [255, 160, 255]])
red_range = np.array([[0, 100, 100], [5, 255, 255], [179, 100, 100], [179, 255, 255]])

# Define the size range for white pieces
min_size = 1900
max_size = 5000

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)

# Process each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold the frame based on the color range
        white_mask = cv2.inRange(hsv, white_range[0], white_range[1])
        red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])
        
        # Apply morphological operations to the mask
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)    
        # Find contours of white pieces
        contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw filled circles around white pieces of certain size
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size < area < max_size:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 0), -1)


        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw filled circles around white pieces of certain size
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size < area < max_size:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (255, 255, 0), -1)
            #cv2.circle(frame, center, radius, (0, 255, 0), 4)  # Draw green circles for red pieces    
        # Write the processed frame to output video
        out.write(frame)

        # Display the processed frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release
cap.release()
out.release()

