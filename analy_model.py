import numpy as np

def get_statistics_object_count(predicted_counts):
    # 1. 예측 객체 수 통계
    mean_count = np.mean(predicted_counts)
    var_count = np.var(predicted_counts)
    std_count = np.std(predicted_counts)

    # 2. 프레임 간 변화율 (ΔN)
    delta_counts = [abs(predicted_counts[i+1] - predicted_counts[i]) for i in range(len(predicted_counts)-1)]
    mean_delta = np.mean(delta_counts)
    std_delta = np.std(delta_counts)

    # 3. 히트맵 기반 예측 밀도 분석
    # 예: 모델에서 추론된 히트맵 배열 리스트 (프레임당 2D 배열)
    heatmaps = [np.random.rand(64, 64) for _ in range(len(predicted_counts))]  # 실제 히트맵 리스트로 교체
    heatmap_total_intensities = [np.sum(hm) for hm in heatmaps]
    heatmap_mean_intensities = [np.mean(hm) for hm in heatmaps]

    avg_total_intensity = np.mean(heatmap_total_intensities)
    avg_mean_intensity = np.mean(heatmap_mean_intensities)

    print("[1] 객체 수 통계")
    print(f"  - 평균 객체 수: {mean_count:.2f}")
    print(f"  - 표준편차: {std_count:.2f}")
    print("\n[2] 프레임 간 변화율")
    print(f"  - 평균 변화량 (ΔN): {mean_delta:.2f}")
    print(f"  - 변화량 표준편차: {std_delta:.2f}")
    print("\n[3] 히트맵 기반 밀도")
    print(f"  - 히트맵 총합 평균: {avg_total_intensity:.2f}")
    print(f"  - 히트맵 평균 강도: {avg_mean_intensity:.2f}")