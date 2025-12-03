from  ultralytics import YOLO
import cv2
import random 
import numpy as np
import os 
print("Loading model...")
model = YOLO("yolo_weights/yolov8n.pt")
