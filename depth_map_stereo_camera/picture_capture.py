import cv2

# Define the path to the left and right videos
left_video_path = 'left.avi'
right_video_path = 'right.avi'

# Open the video capture for both left and right videos
cap_left = cv2.VideoCapture(left_video_path)
cap_right = cv2.VideoCapture(right_video_path)

# Check if the videos are opened correctly
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: Could not open video files.")
    exit()

# Get the frame rate (FPS) of the video
fps_left = cap_left.get(cv2.CAP_PROP_FPS)
fps_right = cap_right.get(cv2.CAP_PROP_FPS)

# Choose the time in seconds where you want to extract the frame
target_time_sec = 33

# Calculate the frame number based on time
frame_number_left = int(fps_left * target_time_sec)
frame_number_right = int(fps_right * target_time_sec)

# Set the position of the video capture to the specific frame number
cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_number_left)
cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_number_right)

# Read the specific frame from both videos
ret_left, frame_left = cap_left.read()
ret_right, frame_right = cap_right.read()

# Check if the frames were read successfully
if not ret_left or not ret_right:
    print("Error: Could not read frames.")
    exit()

# Show the extracted frames
cv2.imshow("Left Frame at 0:52", frame_left)
cv2.imshow("Right Frame at 0:52", frame_right)

# Optionally, save the frames as images
cv2.imwrite('left_chessboard.jpg', frame_left)
cv2.imwrite('right_chessboard.jpg', frame_right)

# Wait for the user to press a key, then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video captures
cap_left.release()
cap_right.release()
