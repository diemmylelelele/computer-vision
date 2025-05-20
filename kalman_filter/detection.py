import cv2
import numpy as np
import time
import os
import csv

'''
The program aims to estimate the error variance of pupil detection for left and right eyes
'''

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Fixed "true" positions of the pupil
true_pupil_position_left = (380, 223)  
true_pupil_position_right = (490, 220) 

# Variables to store detected pupil positions for both eyes
detected_position_left = None
detected_position_right = None

# Variables to store error distances for both eyes
errors_left_eye = []
errors_right_eye = []

prev_time = time.time()

# Create a CSV file to store the detected positions
csv_file = 'pupil_positions.csv'
header = ['Frame', 'Detected_X', 'Detected_Y', 'True_X', 'True_Y', 'Error', 'Eye']

# Write header to CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret or frame is None:  # Ensure frame is valid
        print("Error: Could not capture frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(55, 55))
        
        # Iterate over both detected eyes
        for idx, (ex, ey, ew, eh) in enumerate(eyes[:2]): 
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Apply thresholding to isolate the pupil
            _, thresh = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)

            # Find contours and get the shape of the eye
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                if radius < ew // 4:  # Ensure the detected circle is within the eye region
                    # Draw the pupil center (in red for visualization)
                    # cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
                    cv2.circle(eye_color, (int(cx), int(cy)), 2 , (0, 255, 0), 2)
                    
                    # Record the detected position for the current eye
                    if idx == 0:  # Left eye
                        detected_position_left = (int(cx) , int(cy) )
                        # Calculate the error as the distance between the true position and the detected position
                        error_left = np.sqrt((detected_position_left[0] - true_pupil_position_left[0])**2 +
                                            (detected_position_left[1] - true_pupil_position_left[1])**2)
                        # Save the error for left eye
                        errors_left_eye.append(error_left)
                    else:  # Right eye
                        detected_position_right = (int(cx), int(cy))
                        # Calculate the error as the distance between the true position and the detected position
                        error_right = np.sqrt((detected_position_right[0] - true_pupil_position_right[0])**2 +
                                            (detected_position_right[1] - true_pupil_position_right[1])**2)
                        # Save the error for right eye
                        errors_right_eye.append(error_right)

                    # Write the data to the CSV file for both eyes
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        if idx == 0:  # Left eye
                            writer.writerow([time.time(), detected_position_left[0], detected_position_left[1],
                                             true_pupil_position_left[0], true_pupil_position_left[1],
                                             error_left, 'Left'])
                        else:  # Right eye
                            writer.writerow([time.time(), detected_position_right[0], detected_position_right[1],
                                             true_pupil_position_right[0], true_pupil_position_right[1],
                                             error_right, 'Right'])

    # Calculate frame rate
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw the fixed "true" positions of the pupils (green circles)
    cv2.circle(frame, true_pupil_position_left, 5, (0, 255, 0), -1)
    cv2.circle(frame, true_pupil_position_right, 5, (0, 255, 0), -1)

    # Display the result
    cv2.imshow('Pupil Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Calculate the error variance for both eyes
error_variance_left = np.var(errors_left_eye)
error_variance_right = np.var(errors_right_eye)
print(f"Error Variance Left Eye: {error_variance_left}")
print(f"Error Variance Right Eye: {error_variance_right}")

# Error Variance Left Eye: 4.4403233143261716
# Error Variance Right Eye: 3.534568831897063



