from autodistill_yolov8 import YOLOv8

target_model = YOLOv8("yolov8n-seg.pt")
target_model.train("segmentation_dataset/data.yaml", epochs=37)

