import cv2
from ultralytics import YOLO
import threading
import queue
import time

frame_queue = queue.Queue(timeout=2)
result_queue = queue.Queue(timeout=2)

def capture_threading(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break