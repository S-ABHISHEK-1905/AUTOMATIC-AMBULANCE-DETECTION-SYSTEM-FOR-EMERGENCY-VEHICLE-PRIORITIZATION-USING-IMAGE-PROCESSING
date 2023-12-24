from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.yaml")

results = model.train(data="config.yaml", epochs=30)