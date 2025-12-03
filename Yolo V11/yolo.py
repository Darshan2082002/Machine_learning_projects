from ultralytics import YOLO
model = YOLO("yolo11n.yaml")

model = YOLO("yolo11n.pt")

results = model.train(data="coco8.yaml", epochs=3)
results = model.val()
results = model("https://ultralytics.com/images/bus.jpg")
success = model.export(format="onnx")
