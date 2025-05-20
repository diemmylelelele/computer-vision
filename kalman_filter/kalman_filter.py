import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Fixed "true" positions of the pupil (can be set manually or using the first frame)
true_pupil_position_left = (385, 223)
true_pupil_position_right = (482, 220)

# Initialize Kalman filter for both eyes
kalman_left = cv2.KalmanFilter(4, 2)
kalman_right = cv2.KalmanFilter(4, 2)

# State transition matrix (A)
dt = 1.0  # Time step (1 frame)
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=np.float32)

# Measurement matrix (H)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]], dtype=np.float32)

# Process noise covariance matrix (Q)
Q = np.array([[1e-2, 0, 0, 0],
              [0, 1e-2, 0, 0],
              [0, 0, 1e-2, 0],
              [0, 0, 0, 1e-2]], dtype=np.float32)

# Measurement noise covariance matrix (R) from the error variance
R_left = np.array([[4.44, 0], [0, 4.44]], dtype=np.float32)  
R_right = np.array([[3.53, 0], [0, 3.53]], dtype=np.float32)

# Initialize Kalman filters with state and covariance matrices
kalman_left.transitionMatrix = A
kalman_right.transitionMatrix = A
kalman_left.measurementMatrix = H
kalman_right.measurementMatrix = H
kalman_left.processNoiseCov = Q
kalman_right.processNoiseCov = Q

errors_left = []
errors_right = []

# Set initial state estimate (guess based on true positions)
kalman_left.statePre = np.array([true_pupil_position_left[0], true_pupil_position_left[1], 0, 0], dtype=np.float32)
kalman_right.statePre = np.array([true_pupil_position_right[0], true_pupil_position_right[1], 0, 0], dtype=np.float32)

# Start timer for FPS calculation
prev_time = time.time()

# Set up VideoWriter to save the video
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the correct frame rate from the webcam
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video output (XVID)
output_video = cv2.VideoWriter('tracking_output.avi', fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

# OpenCV loop for tracking
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(55, 55))

        for idx, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Apply thresholding to isolate the pupil
            _, thresh = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

                if radius < ew // 4:
                    if idx == 0:  # Left eye
                        detected_position_left = (int(cx), int(cy))
                        measurement_left = np.array([detected_position_left[0], detected_position_left[1]], dtype=np.float32)

                        # Predict the next state (before the measurement is used)
                        prediction_left = kalman_left.predict()
                        predicted_position_left = (int(prediction_left[0]), int(prediction_left[1]))
                        
                        # Correct the prediction with the measurement
                        estimated_left = kalman_left.correct(measurement_left)
                        estimated_position_left = (int(estimated_left[0]), int(estimated_left[1]))
                        
                        # Calculate the error
                        error_left = np.linalg.norm(estimated_position_left - measurement_left)
                        errors_left.append(error_left)
                        
                        # Draw the predicted (estimated) position in blue
                        cv2.circle(eye_color, estimated_position_left, 2, (255, 0, 0), 2)  # Blue for predicted
                        cv2.putText(frame, "Estimated", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # Draw the measured (detected) position in green
                        cv2.circle(eye_color, detected_position_left, 2, (0, 255, 0), 2)  # Green for detected
                        cv2.putText(frame, "Detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    else:  # Right eye
                        detected_position_right = (int(cx), int(cy))
                        measurement_right = np.array([detected_position_right[0], detected_position_right[1]], dtype=np.float32)

                        # Predict the next state (before the measurement is used)
                        prediction_right = kalman_right.predict()
                        predicted_position_right = (int(prediction_right[0]), int(prediction_right[1]))
                        
                        # Correct the prediction with the measurement
                        estimated_right = kalman_right.correct(measurement_right)
                        estimated_position_right = (int(estimated_right[0]), int(estimated_right[1]))
                        
                        # Calculate the error
                        error_right = np.linalg.norm(estimated_position_right - measurement_right)
                        errors_right.append(error_right)
                        
                        # Draw the predicted (estimated) position in blue
                        cv2.circle(eye_color, estimated_position_right, 2, (255, 0, 0), 2)  # Blue for predicted
                        #cv2.putText(frame, "Right Eye Estimated", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # Draw the measured (detected) position in green
                        cv2.circle(eye_color, detected_position_right, 2, (0, 255, 0), 2)  # Green for detected
                        #cv2.putText(frame, "Right Eye Detected", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    # Calculate FPS (frames per second)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Write the frame to the video output
    output_video.write(frame)
    
    # Display the result
    cv2.imshow('Pupil Detection with Kalman Filter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

if errors_left:
    avg_error_left = sum(errors_left) / len(errors_left)
    print(f"Average tracking error for left eye: {avg_error_left:.2f} pixels")

if errors_right:
    avg_error_right = sum(errors_right) / len(errors_right)
    print(f"Average tracking error for right eye: {avg_error_right:.2f} pixels")
