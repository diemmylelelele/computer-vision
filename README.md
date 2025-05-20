# Computer Vision

This repository presents a curated collection of Computer Vision algorithms and techniques implemented in Python using Jupyter Notebooks and OpenCV. It serves as a hands-on resource for exploring essential and advanced image processing tasks with real-world applications.

### Implemented Techniques

1. ArUco Marker Detection (```aruco_marker/```)
   - Detects ArUco markers in images or videos.
   - Computes homography transformation for marker-based augmentation.
   - Applications: Augmented reality (AR), camera calibration.

2. Hough Transform (```hough_transform/```)
   - Implements Hough Transform to detect lines and circles.
   - Used for edge detection and shape recognition.
   - Applications: Traffic sign detection, medical imaging.

3. Pupil Eye Detection (```pupil_eye_detection/```)
   - Detects pupil position in human eyes.
   - Uses image thresholding and contour detection.
   - Applications: Eye-tracking, medical diagnostics, human-computer interaction.

4. Pupil Eye tracking (```kalman_filter/```)
   - Tracks the pupil center using Kalman filtering for robust and smooth position estimation.
   - Combines image processing with dynamic filtering for better accuracy.
   - Applications: Real-time gaze tracking, attention monitoring, research in human behavior.

5. Depth-map from stereo camera (```depth_map_stereo_camera/```)
   - Computes a depth map from stereo image pairs.
   - Estimates object distance from the camera and triggers a warning if an object is too close.
   - Applications: Autonomous driving, 3D reconstruction, robotics vision systems.
   

### Acknowledgement
Professor Phung M. Duong from Fulbright University Vietnam