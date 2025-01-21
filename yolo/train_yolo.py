from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(data='./tmp.yaml' , epochs=50)