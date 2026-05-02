# YOLOv8m Head Detection — Model Development

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8m-Ultralytics-FF6B35)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia)
![mAP50](https://img.shields.io/badge/mAP50-0.9247-brightgreen)

**Transfer learning training workspace for the head detection model used in [indoor-congestion-analysis](https://github.com/KNUT-Capston-DongJunSu/An-Empty-Seat).**

COCO-pretrained YOLOv8m을 AI-Hub 실내외 군중 특성 데이터셋으로 전이학습하여 실내 공간의 사람 머리를 탐지하는 모델을 개발합니다. 학습 실험, 성능 검증, 최적 confidence 탐색, 데이터셋 변환 도구를 포함합니다.

---

## Final Model Performance

AI-Hub 데이터셋(train/val = 8:2)으로 전이학습한 YOLOv8m의 최종 검증 성능:

| Metric | Score |
|--------|-------|
| Precision | 0.908 |
| Recall | 0.907 |
| **mAP50** | **0.9247** |
| mAP50-95 | 0.6768 |
| Optimal conf (F1 기준) | **0.50** (F1: 0.886) |

> 학습 실험은 `train_result/` 하위 `train5`, `train20`, `train27`, `train34`, `train35`, `train36` 순서로 진행하였으며, 최종 모델은 `train36` 결과를 채택하였습니다.

---

## Project Structure

```
capdi-yolo-dev/
├── modules/
│   ├── yolo_manager.py       # YOLOv8 학습·추론 래퍼 클래스
│   ├── image_inference.py    # 단일 이미지 추론
│   ├── preprocessor.py       # 프레임 전처리
│   ├── processing_results.py # 탐지 결과 후처리
│   └── transformer.py        # 데이터 변환 유틸리티
├── config/
│   ├── custom.yaml           # AI-Hub 데이터셋 경로 설정
│   └── scut.yaml             # SCUT-HEAD 데이터셋 경로 설정
├── train_result/             # 학습 실험 결과 폴더 (train5 ~ train36)
├── train_model.py            # 모델 학습 실행
├── val_models.py             # 모델 성능 검증
├── find_optimal_conf.py      # 최적 confidence threshold 탐색
├── analy_model.py            # 추론 결과 통계 분석
├── json2txt.py               # AI-Hub ODGT(JSON) → YOLO txt 변환
├── xml2txt.py                # XML 어노테이션 → YOLO txt 변환
├── valid_label.py            # 라벨 유효성 검사
├── test_image.py             # 단일 이미지 추론 테스트
└── requirements.txt
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `train_model.py` | YOLOv8m 전이학습 실행 (epoch, batch, patience 등 설정 가능) |
| `val_models.py` | 학습된 모델의 mAP50, mAP50-95, Precision, Recall 출력 |
| `find_optimal_conf.py` | conf 0.1–0.9 구간을 sweep하여 F1 기준 최적 threshold 탐색 |
| `analy_model.py` | 프레임별 객체 수 통계 (평균, 분산, 프레임 간 변화율) |
| `json2txt.py` | AI-Hub ODGT JSON 포맷을 YOLO txt 라벨로 변환 |
| `xml2txt.py` | XML 어노테이션을 YOLO txt 라벨로 변환 |
| `valid_label.py` | 생성된 라벨 파일 유효성 검사 |
| `test_image.py` | 단일 이미지에 대한 추론 결과 시각화 |

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
# CUDA 11.8 + PyTorch 2.4.1 환경 권장
```

### Dataset Setup

`config/custom.yaml`에서 데이터셋 경로를 수정합니다:

```yaml
train: ../datasets/images/train
val:   ../datasets/images/val
nc: 1
names: ['person']
```

AI-Hub 데이터 사용 시 JSON 라벨을 YOLO 포맷으로 변환합니다:

```bash
python json2txt.py
```

### Training

```bash
python train_model.py
```

### Validation

```bash
python val_models.py
```

### Find Optimal Confidence Threshold

```bash
python find_optimal_conf.py
# 출력 예시: 최적 conf: 0.50 (F1 Score: 0.886)
```

---

## Datasets Used

| Dataset | Format | Purpose |
|---------|--------|---------|
| AI-Hub 실내외 군중 특성 데이터 | JSON (ODGT) | 최종 전이학습 (채택) |
| SCUT-HEAD | XML | 비교 실험 |

---

## Related

- **Main system**: [indoor-congestion-analysis](https://github.com/KNUT-Capston-DongJunSu/An-Empty-Seat) — 이 레포에서 학습된 모델이 실제 서비스에 사용됩니다.
- Capstone Design 2025, Department of Electronic Engineering, KNUT
  - Advisor: Prof. Song Chang-ik
  - Team DongJunSu: Kim Dong-in (2222007), Kim Jun-hee (2122014)
