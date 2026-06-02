from ultralytics import YOLO

model = YOLO('/home/hzh/container_yolo/yolo11m.pt')

model.train(
    data='/home/hzh/container_yolo/merged_container_number_yolo_b_rf/data.yaml',
    epochs=100,
    imgsz=960,
    batch=32,
    device='0,1,2,3',
    workers=16,
    project='/home/hzh/container_yolo/runs',
    name='region_b_rf_yolo11m_960_100e',
    exist_ok=True,
    patience=30,
)
