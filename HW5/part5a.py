#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# set the number of frames to keep in motion history
t = 20

# create an empty list to store the last t binary masks
motion_history = []

# read the input video
cap = cv2.VideoCapture('p5_video2.m4v')

# get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# create the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video2.mp4', fourcc, fps, (width, height), isColor=True)

# read the first frame as the background model
ret, bg = cap.read()
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    # read the current frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # compute the absolute difference between the current frame and the background model
    diff = cv2.absdiff(gray, bg_gray)
    
    # apply thresholding to create a binary mask
    _, mask = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
    
    # apply morphological operations to remove noise and fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # add the binary mask to the list of last t masks
    motion_history.append(mask)
    
    # if the list has more than t masks, remove the oldest mask
    if len(motion_history) > t:
        motion_history.pop(0)
    
    # create the motion history image by combining the last t masks
    motion_history_image = np.zeros((height, width), dtype=np.uint8)
    for i, mask in enumerate(motion_history):
        # set the pixel value based on the age of the motion
        pixel_value = int(255 * (i+1) / t)
        # set the pixel color based on the pixel value
        pixel_color = (255-pixel_value, 255-pixel_value, 255-pixel_value)
        # apply the mask and color to the motion history image
        motion_history_image[mask == 255] = pixel_value
    
    # convert the motion history image to color and write it to the output video
    motion_history_image_color = cv2.cvtColor(motion_history_image, cv2.COLOR_GRAY2BGR)
    out.write(motion_history_image_color)

    
    # update the background model
    bg_gray = gray
    
    # display the output video
    cv2.imshow('motion history', motion_history_image_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
out.release()
cv2.destroyAllWindows()

