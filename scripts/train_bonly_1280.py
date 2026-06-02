from ultralytics import YOLO

model = YOLO("/home/hzh/container_yolo/yolo11m.pt")

model.train(
    data="/home/hzh/container_yolo/container_v1i_yolov11_b/data_fixed.yaml",
    epochs=100,
    imgsz=1280,
    batch=16,
    device="0,1,2,3",
    workers=16,
    project="/home/hzh/container_yolo/runs",
    name="region_b_yolo11m_1280_100e",
    exist_ok=True,
    patience=30,
)
