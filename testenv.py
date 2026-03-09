from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.predict("https://ultralytics.com/images/bus.jpg", show=True)