#!/usr/bin/env python
# coding: utf-8

# In[132]:


import cv2
import numpy as np

# Define the video capture object
cap = cv2.VideoCapture('p5_video1.m4v')

# Define the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the kernel for morphological operations
kernel = np.ones((7,7),np.uint8)

# Define the minimum area for a contour to be considered a piece
min_area = 4000

# Create the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video1b.mp4', fourcc, fps, (width, height), isColor=True)

# Define the color of the piece trajectories
color = (255, 0, 0)

# Define the thickness of the piece trajectories
thickness = 2

# Define a dictionary to store the previous centers of the pieces
prev_centers = {}

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to the frame
    fgmask = fgbg.apply(frame)

    # Apply morphological opening to the foreground mask
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find the contours in the foreground mask
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for i, contour in enumerate(contours):
        # Compute the area of the contour
        area = cv2.contourArea(contour)

        # If the contour is large enough to be a piece, track its movement
        if area > min_area:
            # Compute the bounding box of the contour
            x,y,w,h = cv2.boundingRect(contour)

            # Compute the center of the bounding box
            cx, cy = x + w // 2, y + h // 2

            # Draw a circle at the center of the bounding box
            cv2.circle(frame, (cx, cy), 5, color, -1)

            # Connect the center to the previous center if it exists
            if i in prev_centers:
                prev_center = prev_centers[i]
                cv2.line(frame, prev_center, (cx, cy), color, thickness)

            # Update the previous center for this piece
            prev_centers[i] = (cx, cy)

    # Show the output video
    cv2.imshow('Board Game', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write the frame to the output video
    out.write(frame)

    # Clear the previous centers dictionary for pieces that were not detected in this frame
    prev_centers = {k: v for k, v in prev_centers.items() if k in range(len(contours))}

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


# In[133]:


import cv2
import numpy as np

# Define the video capture object
cap = cv2.VideoCapture('p5_video2.m4v')

# Define the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the kernel for morphological operations
kernel = np.ones((7,7),np.uint8)

# Define the minimum area for a contour to be considered a piece
min_area = 4000

# Create the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video2b.mp4', fourcc, fps, (width, height), isColor=True)

# Define the color of the piece trajectories
color = (255, 0, 0)

# Define the thickness of the piece trajectories
thickness = 2

# Define a dictionary to store the previous centers of the pieces
prev_centers = {}

while(cap.isOpened()):
    # Read the next frame from the video
    ret, frame = cap.read()
    if ret==True:
    

        # Apply background subtraction to the frame
        fgmask = fgbg.apply(frame)

        # Apply morphological opening to the foreground mask
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Find the contours in the foreground mask
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours
        for i, contour in enumerate(contours):
            # Compute the area of the contour
            area = cv2.contourArea(contour)

            # If the contour is large enough to be a piece, track its movement
            if area > min_area:
                # Compute the bounding box of the contour
                x,y,w,h = cv2.boundingRect(contour)

                # Compute the center of the bounding box
                cx, cy = x + w // 2, y + h // 2

                # Draw a circle at the center of the bounding box
                cv2.circle(frame, (cx, cy), 5, color, -1)

                # Connect the center to the previous center if it exists
                if i in prev_centers:
                    prev_center = prev_centers[i]
                    cv2.line(frame, prev_center, (cx, cy), color, thickness)

                # Update the previous center for this piece
                prev_centers[i] = (cx, cy)

        # Show the output video
        cv2.imshow('Board Game', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write the frame to the output video
        out.write(frame)

        # Clear the previous centers dictionary for pieces that were not detected in this frame
        prev_centers = {k: v for k, v in prev_centers.items() if k in range(len(contours))}
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


# In[136]:


import cv2
import numpy as np

# Define the video capture object
cap = cv2.VideoCapture('p5_video1.m4v')

# Define the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the kernel for morphological operations
kernel = np.ones((7,7),np.uint8)

# Define the minimum area for a contour to be considered a piece
min_area = 4000

# Create the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video1b.mp4', fourcc, fps, (width, height), isColor=True)

# Define the color of the piece trajectories
color = (255, 0, 0)

# Define the thickness of the piece trajectories
thickness = 2

# Define a dictionary to store the previous centers of the pieces
prev_centers = {}

while(cap.isOpened()):
    # Read the next frame from the video
    ret, frame = cap.read()
    if ret==True:
    

        # Apply background subtraction to the frame
        fgmask = fgbg.apply(frame)

        # Apply morphological opening to the foreground mask
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Find the contours in the foreground mask
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours
        for i, contour in enumerate(contours):
            # Compute the area of the contour
            area = cv2.contourArea(contour)

            # If the contour is large enough to be a piece, track its movement
            if area > min_area:
                # Compute the bounding box of the contour
                x,y,w,h = cv2.boundingRect(contour)

                # Compute the center of the bounding box
                cx, cy = x + w // 2, y + h // 2

                # Draw a circle at the center of the bounding box
                cv2.circle(frame, (cx, cy), 5, color, -1)

                # Connect the center to the previous center if it exists
                if i in prev_centers:
                    prev_center = prev_centers[i]
                    cv2.line(frame, prev_center, (cx, cy), color, thickness)

                # Update the previous center for this piece
                prev_centers[i] = (cx, cy)

        # Show the output video
        cv2.imshow('Board Game', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write the frame to the output video
        out.write(frame)

        # Clear the previous centers dictionary for pieces that were not detected in this frame
        prev_centers = {k: v for k, v in prev_centers.items() if k in range(len(contours))}
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


# In[137]:


import cv2
import numpy as np

# Define the video capture object
cap = cv2.VideoCapture('p5_video1.m4v')

# Define the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the kernel for morphological operations
kernel = np.ones((7,7),np.uint8)

# Define the minimum area for a contour to be considered a piece
min_area = 4000

# Create the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video1b.mp4', fourcc, fps, (width, height), isColor=True)

# Define the color of the piece trajectories
color = (255, 0, 0)

# Define the thickness of the piece trajectories
thickness = 2

# Define a dictionary to store the previous centers of the pieces
prev_centers = {}

while(cap.isOpened()):
    # Read the next frame from the video
    ret, frame = cap.read()
    if ret==True:
    

        # Apply background subtraction to the frame
        fgmask = fgbg.apply(frame)

        # Apply morphological opening to the foreground mask
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Find the contours in the foreground mask
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours
        for i, contour in enumerate(contours):
            # Compute the area of the contour
            area = cv2.contourArea(contour)

            # If the contour is large enough to be a piece, track its movement
            if area > min_area:
                # Compute the bounding box of the contour
                x,y,w,h = cv2.boundingRect(contour)

                # Compute the center of the bounding box
                cx, cy = x + w // 2, y + h // 2

                # Draw a circle at the center of the bounding box
                cv2.circle(frame, (cx, cy), 5, color, -1)

                # Connect the center to the previous center if it exists
                if i in prev_centers:
                    prev_center = prev_centers[i]
                    cv2.line(frame, prev_center, (cx, cy), color, thickness)

                # Update the previous center for this piece
                prev_centers[i] = (cx, cy)

        # Show the output video
        cv2.imshow('Board Game', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write the frame to the output video
        out.write(frame)

        # Clear the previous centers dictionary for pieces that were not detected in this frame
        prev_centers = {k: v for k, v in prev_centers.items() if k in range(len(contours))}
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

