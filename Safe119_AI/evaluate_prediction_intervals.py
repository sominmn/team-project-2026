
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"

df = pd.read_csv(
    DATA / "advanced_model_predictions.csv",
    encoding="utf-8-sig"
)

MODEL = "RF 잔차 적응형"
TARGET_COVERAGE = 0.90

df = df[df["모델"] == MODEL].copy()

df["날짜"] = pd.to_datetime(
    dict(
        year=df["평가연도"],
        month=df["월"],
        day=1
    )
)

df = df.sort_values(
    ["날짜", "지역"]
).reset_index(drop=True)

df["절대오차"] = (
    df["실제건수"] - df["예측건수"]
).abs()

def conformal_quantile(errors, coverage=0.90):
    values = np.sort(np.asarray(errors, dtype=float))
    n = len(values)

    if n == 0:
        raise ValueError("보정 오차 데이터가 없습니다.")

    # 유한 표본 보정 분위수
    rank = int(np.ceil((n + 1) * coverage))

    if rank > n:
        return np.inf

    return float(values[rank - 1])

results = []
details = []

# 평가연도보다 과거인 연도만 보정에 사용
for year in [2023, 2024, 2025]:
    calibration = df[df["평가연도"] == year - 1]
    test = df[df["평가연도"] == year].copy()

    if calibration.empty or test.empty:
        continue

    q = conformal_quantile(
        calibration["절대오차"],
        TARGET_COVERAGE
    )

    test["구간하한"] = np.maximum(
        0,
        test["예측건수"] - q
    )
    test["구간상한"] = test["예측건수"] + q

    test["포함여부"] = (
        (test["실제건수"] >= test["구간하한"])
        & (test["실제건수"] <= test["구간상한"])
    )

    test["구간폭"] = (
        test["구간상한"] - test["구간하한"]
    )

    coverage = test["포함여부"].mean() * 100

    results.append({
        "평가연도": year,
        "보정연도": year - 1,
        "보정표본수": len(calibration),
        "보정오차분위수": round(q, 2),
        "실제포함률(%)": round(coverage, 2),
        "평균구간폭": round(test["구간폭"].mean(), 2)
    })

    details.append(test)

    print(f"\n===== {year}년 예측 구간 평가 =====")
    print(f"보정 데이터: {year - 1}년")
    print(f"보정 표본 수: {len(calibration)}")
    print(f"구간 반폭: {q:.2f}건")
    print(f"실제 포함률: {coverage:.2f}%")
    print(
        f"평균 구간폭: "
        f"{test['구간폭'].mean():.2f}건"
    )

summary = pd.DataFrame(results)
all_details = pd.concat(details, ignore_index=True)

regional = (
    all_details.groupby(["평가연도", "지역"])
    .agg(
        포함률=("포함여부", "mean"),
        평균구간폭=("구간폭", "mean"),
        평가건수=("포함여부", "size")
    )
    .reset_index()
)

regional["포함률"] *= 100

summary.to_csv(
    DATA / "prediction_interval_results.csv",
    index=False,
    encoding="utf-8-sig"
)

all_details.to_csv(
    DATA / "prediction_interval_details.csv",
    index=False,
    encoding="utf-8-sig"
)

regional.to_csv(
    DATA / "prediction_interval_regional.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 예측 구간 최종 요약 =====")
print(summary.to_string(index=False))

print("\n===== 지역별 포함률 하위 10개 =====")
print(
    regional.sort_values("포함률")
    .head(10)
    .round(2)
    .to_string(index=False)
)

print("\n예측 구간 검증 완료!")
