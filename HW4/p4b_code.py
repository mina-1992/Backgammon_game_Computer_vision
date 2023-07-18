#!/usr/bin/env python
# coding: utf-8

# In[102]:


import cv2
import numpy as np

# Read the input video file
cap = cv2.VideoCapture('p4_video1.m4v')

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video1b.mp4', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

# Define HSV ranges for white and red circles
white_range = np.array([[0, 0, 157], [255, 150, 255]])
red_range = np.array([[0, 100, 100], [5, 255, 255], [179, 100, 100], [179, 255, 255]])

# Process each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # Define starting point for first line
        start_point = (65, 0)

        # Define distance between lines
        distance = 145

        # Convert the image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the image to get binary images of white and red circles
        white_mask = cv2.inRange(hsv, white_range[0], white_range[1])
        red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])


        # Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convert to grayscale
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 20, 100)

        # Hough Circle Transform
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=16, param1=7, param2= 70 , minRadius=0, maxRadius=68)

        contours_white, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        num_circles_top = []
        num_circles_bottom = []

        num_circles_top_white = []
        num_circles_bottom_white = []

        circles = np.round(circles[0, :]).astype("int")

        for i in range(frame.shape[1]//distance):
            x1 = i * distance
            x2 = (i+1) * distance# Draw detected circles
            if circles is not None:
                circle_count_bottom_red = 0
                circle_count_top_red = 0
                circle_count_top_white = 0
                circle_count_bottom_white = 0

                for (x, y, r) in circles:
                    if x >= x1 and x < x2: 
                        if y < frame.shape[0]//2:
                            if white_mask[y][x] == 255:
                                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                                circle_count_top_white += 1
                            else:
                                cv2.circle(frame, (x, y), r, (255, 0, 255), 2)
                                circle_count_top_red += 1
                        else:            
                            if red_mask[y][x] == 255:
                                cv2.circle(frame, (x, y), r, (255, 0, 255), 2)
                                circle_count_bottom_red += 1
                            else: 
                                circle_count_bottom_white += 1
                                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)



            for i in range (circle_count_top_white > 0):    
                num_circles_top_white.append(circle_count_top_white)
                cv2.putText(frame, "W" + str(circle_count_top_white), (x1+105, frame.shape[0]//2 -90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  

            for i in range (circle_count_bottom_white > 0):
                num_circles_bottom_white.append(circle_count_bottom_white)      
                cv2.putText(frame, "W" + str(circle_count_bottom_white ), (x1+105, frame.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
            for i in range( circle_count_bottom_red > 0 ):
                num_circles_bottom.append(circle_count_bottom_red)
                cv2.putText(frame, "R" + str(circle_count_bottom_red), (x1+105, frame.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            for i in range(  circle_count_top_red > 0 ):
                num_circles_top.append(circle_count_top_red)
                cv2.putText(frame, "R" + str(circle_count_top_red), (x1+105, frame.shape[0]//2 -90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)   
            
            for i in range(circle_count_bottom_red == 0 and circle_count_top_red == 0 and circle_count_bottom_white == 0 and circle_count_top_white == 0):
                
                num_circles_top.append(0)
                num_circles_bottom.append(0)
                num_circles_bottom_white.append(0)
                num_circles_top_white.append(0)

                cv2.putText(frame, "0", (x1+110, frame.shape[0]//2 - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, "0", (x1+110, frame.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 255), 3)

        # Write the frame to the output video file
        out.write(frame)
    
        # Display the resulting frame
        cv2.imshow('frame',frame)
    
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()



# In[ ]:


import cv2
import numpy as np

# Read the input video file
cap = cv2.VideoCapture('p4_video2.m4v')

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video2b.mp4', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

# Define HSV ranges for white and red circles
white_range = np.array([[0, 0, 157], [255, 150, 255]])
red_range = np.array([[0, 100, 100], [5, 255, 255], [179, 100, 100], [179, 255, 255]])

# Process each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # Define starting point for first line
        start_point = (65, 0)

        # Define distance between lines
        distance = 145

        # Convert the image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the image to get binary images of white and red circles
        white_mask = cv2.inRange(hsv, white_range[0], white_range[1])
        red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])


        # Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convert to grayscale
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 20, 100)

        # Hough Circle Transform
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=16, param1=7, param2= 70 , minRadius=0, maxRadius=68)

        contours_white, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        num_circles_top = []
        num_circles_bottom = []

        num_circles_top_white = []
        num_circles_bottom_white = []

        circles = np.round(circles[0, :]).astype("int")

        for i in range(frame.shape[1]//distance):
            x1 = i * distance
            x2 = (i+1) * distance# Draw detected circles
            if circles is not None:
                circle_count_bottom_red = 0
                circle_count_top_red = 0
                circle_count_top_white = 0
                circle_count_bottom_white = 0

                for (x, y, r) in circles:
                    if x >= x1 and x < x2: 
                        if y < frame.shape[0]//2:
                            if white_mask[y][x] == 255:
                                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                                circle_count_top_white += 1
                            else:
                                cv2.circle(frame, (x, y), r, (255, 0, 255), 2)
                                circle_count_top_red += 1
                        else:            
                            if red_mask[y][x] == 255:
                                cv2.circle(frame, (x, y), r, (255, 0, 255), 2)
                                circle_count_bottom_red += 1
                            else: 
                                circle_count_bottom_white += 1
                                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)



            for i in range (circle_count_top_white > 0):    
                num_circles_top_white.append(circle_count_top_white)
                cv2.putText(frame, "W" + str(circle_count_top_white), (x1+105, frame.shape[0]//2 -90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  

            for i in range (circle_count_bottom_white > 0):
                num_circles_bottom_white.append(circle_count_bottom_white)      
                cv2.putText(frame, "W" + str(circle_count_bottom_white ), (x1+105, frame.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
            for i in range( circle_count_bottom_red > 0 ):
                num_circles_bottom.append(circle_count_bottom_red)
                cv2.putText(frame, "R" + str(circle_count_bottom_red), (x1+105, frame.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            for i in range(  circle_count_top_red > 0 ):
                num_circles_top.append(circle_count_top_red)
                cv2.putText(frame, "R" + str(circle_count_top_red), (x1+105, frame.shape[0]//2 -90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)   
            
            for i in range(circle_count_bottom_red == 0 and circle_count_top_red == 0 and circle_count_bottom_white == 0 and circle_count_top_white == 0):
                
                num_circles_top.append(0)
                num_circles_bottom.append(0)
                num_circles_bottom_white.append(0)
                num_circles_top_white.append(0)

                cv2.putText(frame, "0", (x1+110, frame.shape[0]//2 - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, "0", (x1+110, frame.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 255), 3)

        # Write the frame to the output video file
        out.write(frame)
    
        # Display the resulting frame
        cv2.imshow('frame',frame)
    
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()



# In[ ]:


import cv2
import numpy as np

# Read the input video file
cap = cv2.VideoCapture('p4_video3.m4v')

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video3b.mp4', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

# Define HSV ranges for white and red circles
white_range = np.array([[0, 0, 157], [255, 150, 255]])
red_range = np.array([[0, 100, 100], [5, 255, 255], [179, 100, 100], [179, 255, 255]])

# Process each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # Define starting point for first line
        start_point = (65, 0)

        # Define distance between lines
        distance = 145

        # Convert the image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the image to get binary images of white and red circles
        white_mask = cv2.inRange(hsv, white_range[0], white_range[1])
        red_mask = cv2.inRange(hsv, red_range[0], red_range[1]) + cv2.inRange(hsv, red_range[2], red_range[3])


        # Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convert to grayscale
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 20, 100)

        # Hough Circle Transform
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=16, param1=7, param2= 70 , minRadius=0, maxRadius=68)

        contours_white, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        num_circles_top = []
        num_circles_bottom = []

        num_circles_top_white = []
        num_circles_bottom_white = []

        circles = np.round(circles[0, :]).astype("int")

        for i in range(frame.shape[1]//distance):
            x1 = i * distance
            x2 = (i+1) * distance# Draw detected circles
            if circles is not None:
                circle_count_bottom_red = 0
                circle_count_top_red = 0
                circle_count_top_white = 0
                circle_count_bottom_white = 0

                for (x, y, r) in circles:
                    if x >= x1 and x < x2: 
                        if y < frame.shape[0]//2:
                            if white_mask[y][x] == 255:
                                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                                circle_count_top_white += 1
                            else:
                                cv2.circle(frame, (x, y), r, (255, 0, 255), 2)
                                circle_count_top_red += 1
                        else:            
                            if red_mask[y][x] == 255:
                                cv2.circle(frame, (x, y), r, (255, 0, 255), 2)
                                circle_count_bottom_red += 1
                            else: 
                                circle_count_bottom_white += 1
                                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)



            for i in range (circle_count_top_white > 0):    
                num_circles_top_white.append(circle_count_top_white)
                cv2.putText(frame, "W" + str(circle_count_top_white), (x1+105, frame.shape[0]//2 -90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  

            for i in range (circle_count_bottom_white > 0):
                num_circles_bottom_white.append(circle_count_bottom_white)      
                cv2.putText(frame, "W" + str(circle_count_bottom_white ), (x1+105, frame.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
            for i in range( circle_count_bottom_red > 0 ):
                num_circles_bottom.append(circle_count_bottom_red)
                cv2.putText(frame, "R" + str(circle_count_bottom_red), (x1+105, frame.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            for i in range(  circle_count_top_red > 0 ):
                num_circles_top.append(circle_count_top_red)
                cv2.putText(frame, "R" + str(circle_count_top_red), (x1+105, frame.shape[0]//2 -90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)   
            
            for i in range(circle_count_bottom_red == 0 and circle_count_top_red == 0 and circle_count_bottom_white == 0 and circle_count_top_white == 0):
                
                num_circles_top.append(0)
                num_circles_bottom.append(0)
                num_circles_bottom_white.append(0)
                num_circles_top_white.append(0)

                cv2.putText(frame, "0", (x1+110, frame.shape[0]//2 - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, "0", (x1+110, frame.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 255), 3)

        # Write the frame to the output video file
        out.write(frame)
    
        # Display the resulting frame
        cv2.imshow('frame',frame)
    
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()


