#!/usr/bin/env python

"""
HOW TO RUN THIS

cd robotron2084gym/
python3.11 -mvenv .venv
source .venv/bin/activate

PROBABLY NOT:
pip install -r requirements.txt

MAYBE:
pip install opencv-python

python findfps.py


"""

import cv2
import time

if __name__ == '__main__':
    capturing = False

    # Start default camera
    video = cv2.VideoCapture(0)
    video.set(3, 1280)
    video.set(4, 720)

    # Create window to display video
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    # Start timer (for calculating frame rate)
    timer = time.time()

    # Capture a frame, add fps text, and display it
    step = 0
    while True:
        ret, frame = video.read()
        if ret:
            # Calculate fps
            fps = 1.0 / (time.time() - timer)

            # Add fps text to frame
            cv2.putText(frame, f'FPS: {fps:.2f} {capturing and "Capturing Frames." or ""}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Reset timer
            timer = time.time()

            # Display the resulting frame
            cv2.imshow('Video', frame)

            keyPress = cv2.waitKey(1)
            # Press Q on keyboard to  exit
            if keyPress & 0xFF == ord('q'):
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Press C on keyboard to start/stop capturing frames
            if keyPress & 0xFF == ord('c'):
            #if cv2.waitKey(1) & 0xFF == ord('c'):
                capturing = not capturing

            step += 1
            if capturing and step % 25 == 0:
                cv2.imwrite(f'frame_{timer}.jpg', frame)

    # Release video
    video.release()
