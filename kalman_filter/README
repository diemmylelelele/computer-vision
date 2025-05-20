# Kalman Filter for Human Pupil Center Eye Tracking

This project implements a real-time eye-tracking system that detects and tracks the center of the pupils using a webcam. It utilizes OpenCV's Haar cascades for face and eye detection and applies a Kalman filter to smooth and predict pupil positions over time.

###  Features

- Real-time face and eye detection using Haar cascades

- Pupil detection via image thresholding and contour detection

- Application of the Kalman filter for more robust and noise-tolerant pupil tracking

- Visual comparison of detected vs predicted pupil positions

- FPS counter to monitor real-time performance

- Calculation and display of average tracking error for both eyes

### How It Works

1. The system captures frames from the webcam.

2. Haar cascades detect the face and eyes within the frame

3. Within each eye region:

   - Image is thresholded to isolate dark pupil region.

   - Contours are analyzed to locate the pupil.

   - Detected pupil position is measured.

4. A Kalman filter is used to:

   - Predict the next pupil location.

   - Correct the prediction using the detected position.

5. Results are visualized:

   - Green dot: Detected pupil center

   - Blue dot: Predicted (Kalman-filtered) pupil center

6. Average tracking error is computed and printed after quitting the app.

### Installation
Make sure you have Python and the required packages:

```pip install opencv-python numpy```

### Usage
Simply run the script: ```kalman_filter.py```
