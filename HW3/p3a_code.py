#!/usr/bin/env python
# coding: utf-8

# In[81]:


import cv2
import numpy as np

# Read the input video file
cap = cv2.VideoCapture('p3_video2.m4v')

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

# Define the color ranges for white and red pieces in the HSV color space

red_range = np.array([[0, 100, 100], [0, 255, 255], [177, 100, 100], [179, 255, 255]])

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)

# Process each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold the frame based on the color ranges
        red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])
        
        # Apply morphological operations to the masks

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        frame[red_mask != 0] = (255, 0, 0)
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
cap = cv2.VideoCapture('p3_video1.m4v')

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video2.mp4', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

# Define the color ranges for white and red pieces in the HSV color space

red_range = np.array([[0, 100, 100], [0, 255, 255], [177, 100, 100], [179, 255, 255]])

# Apply morphological operations to the masks
kernel = np.ones((5, 5), np.uint8)

# Process each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold the frame based on the color ranges
        red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])
        
        # Apply morphological operations to the masks

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        frame[red_mask != 0] = (255, 0, 0)
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

