import cv2
import os
import time

# Parameters
save_dir_ball = "/Users/research/PycharmProjects/PhDProjects/RadarProcessingProject/dataset/ball"       # folder to save images
num_images = 20           # total number of images to capture
camera_id = 0              # 0 = external webcam, 1 = built in webcamcc

# Open the webcam
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

count = 0
while count < num_images:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Suppose 'frame' is your original 1920x1080 image
    target_size = 256
    h, w, _ = frame.shape

    # Step 1: scale shorter dimension to target_size
    if h < w:
        scale = target_size / h
        new_h = target_size
        new_w = int(w * scale)
    else:
        scale = target_size / w
        new_w = target_size
        new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))  # cv2 uses INTER_LINEAR by default (averaging for downscale)

    # Step 2: center-crop longer dimension to 256
    start_x = (new_w - target_size) // 2
    start_y = (new_h - target_size) // 2
    cropped = resized[start_y:start_y + target_size, start_x:start_x + target_size]

    # Show the live video
    cv2.imshow('Capture', frame)

    # Save the image when you press 'c'
    key = cv2.waitKey(1)
    if key == ord('c'):
        filenameball = os.path.join(save_dir_ball, f'image_ball_{int(time.time())}.jpg')
        cv2.imwrite(filenameball, cropped)
        count += 1

    # Press 'q' to quit early
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()