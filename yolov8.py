from autodistill_yolov8 import YOLOv8

target_model = YOLOv8("yolov8n-seg.pt") # change the "n for tiny" after 8 with s for small, m for medium or l for large
target_model.train("segmentation_dataset/data.yaml", epochs=37)

