# YOLOv8m Head Detection

YOLOv8m으로 사람 머리 탐지 모델을 전이학습하고 평가하는 작업 공간입니다. 여기서 만든 모델이 [indoor-congestion-analysis](https://github.com/jwhitekim/indoor-congestion-analysis)의 혼잡도 분석에 쓰입니다.

![머리마다 박스와 추적 경로가 그려지고 "9 people | Status: Common"이 표시된 매장 CCTV 프레임](docs/images/result-frame.jpg)

<sub>이 모델을 쓴 indoor-congestion-analysis의 결과 영상(`results/supermaket.mp4`)에서 뽑은 프레임입니다.</sub>

| Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- |
| 0.908 | 0.907 | **0.9247** | 0.6768 |

AI-Hub 실내외 군중 데이터(crowd-ah) 기준 기록입니다. 최종 가중치와 이 수치는 indoor-congestion-analysis의 [`models/`](https://github.com/jwhitekim/indoor-congestion-analysis/tree/main/models)에 있고, 이 저장소에는 평가 로그와 가중치가 없습니다.

## 들어 있는 것

| 작업 | 파일 |
| --- | --- |
| 학습 | [`train_model.py`](train_model.py), [`modules/yolo_manager.py`](modules/yolo_manager.py) |
| 평가, confidence 탐색 | [`val_models.py`](val_models.py), [`find_optimal_conf.py`](find_optimal_conf.py) |
| 라벨 변환(JSON·XML → YOLO) | [`json2txt.py`](json2txt.py), [`xml2txt.py`](xml2txt.py), [`modules/transformer.py`](modules/transformer.py) |
| 이미지 추론 | [`modules/image_inference.py`](modules/image_inference.py) |

## 시작하기

[`requirements.txt`](requirements.txt)는 Python 3.8, PyTorch 2.4.1+cu118, Ultralytics 8.3.141을 명시합니다. PyTorch가 CUDA 11.8 빌드로 고정돼 있어 해당 빌드의 설치 경로는 따로 확인해야 합니다.

```bash
pip install -r requirements.txt
```

데이터셋은 포함되어 있지 않습니다. YOLO 형식 이미지·라벨을 준비하고 [`config/custom.yaml`](config/custom.yaml)의 `train`, `val` 경로를 맞춥니다.

> [!WARNING]
> `train_model.py`, `json2txt.py`, `test_image.py`는 저장소에 없는 `core` 모듈을 가져옵니다. `modules` 패키지는 초기화할 때 저장소에 없는 `densEstAI`를 가져오므로 `xml2txt.py` 등 `modules`를 쓰는 코드도 그대로는 실행되지 않습니다. 아래처럼 Ultralytics를 직접 호출하면 같은 설정으로 학습할 수 있습니다.

```python
from ultralytics import YOLO

YOLO("yolov8m.pt").train(
    data="config/custom.yaml", epochs=20, imgsz=640, batch=4,
    optimizer="SGD", lr0=0.01, patience=5, project="results", name="train36",
)
```

`val_models.py`의 기본 실행은 COCO 평가입니다. 머리 탐지 모델은 `val(model_path, data_yaml)`에 학습 가중치와 데이터 설정을 넘겨 평가합니다. `find_optimal_conf.py`가 읽는 `results/train/weights/best.pt`도 실제 학습 결과 경로(`results/train36/...`)에 맞게 바꿔야 합니다.

## 관련

Capstone Design 2025, KNUT 전자공학과. Team DongJunSu(김동인, 김준희), 지도교수 송창익.

## 라이선스

LICENSE 파일이 없습니다. 라이선스는 확인이 필요합니다.
