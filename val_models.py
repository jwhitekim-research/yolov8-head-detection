import os
from ultralytics import YOLO

def val(model_path, data_yaml):
    # 모델 경로 설정
    model = YOLO(model_path)

    # 모델1 성능 평가
    metrics = model.val(data=data_yaml, split="val")
    print("\n[모델 성능]")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

if __name__=='__main__':
    # val('results/medium.pt','config/custom.yaml')

    from ultralytics import settings
    settings.update({'datasets_dir': 'C:/Users/prml-s-j.hong/datasets'})

    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")
    metrics = model.val(data="coco.yaml", split="val")
    print(metrics.box.mp)   # mean Precision
    print(metrics.box.mr)   # mean Recall
    print(metrics.box.map50)
    print(metrics.box.map) 

    

