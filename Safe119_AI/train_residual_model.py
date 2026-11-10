
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(
    DATA_DIR / "ml_features.csv",
    encoding="utf-8-sig"
)

df = df.sort_values(
    ["지역", "연도", "월"]
).reset_index(drop=True)

features = [
    "연도",
    "월",
    "지역",
    "이전달건수",
    "최근3개월평균",
    "작년동월건수"
]

# 이전 달 대비 실제 증가·감소량
df["변화량"] = (
    df["구조활동건수"] - df["이전달건수"]
)

def create_model():
    preprocess = ColumnTransformer([
        (
            "region",
            OneHotEncoder(handle_unknown="ignore"),
            ["지역"]
        )
    ], remainder="passthrough")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

# 보정 강도: 0이면 이전 달 기준,
# 1이면 AI가 예측한 변화량 전체 반영
alphas = [0.25, 0.5, 0.75, 1.0]

results = []
predictions = []

for test_year in range(2022, 2026):
    print(f"\n===== {test_year}년 AI 보정 모델 검증 =====")

    train = df[df["연도"] < test_year].copy()
    test = df[df["연도"] == test_year].copy()

    model = create_model()

    model.fit(
        train[features],
        train["변화량"]
    )

    predicted_change = model.predict(
        test[features]
    )

    baseline = test["이전달건수"].to_numpy()
    actual = test["구조활동건수"].to_numpy()

    # 이전 달 기준 모델
    experiments = [("이전 달 기준", baseline)]

    # 보정 강도별 실험
    for alpha in alphas:
        corrected = np.maximum(
            baseline + alpha * predicted_change,
            0
        )

        experiments.append((
            f"AI 보정 {alpha}",
            corrected
        ))

    for label, pred in experiments:
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(
            mean_squared_error(actual, pred)
        )

        results.append({
            "평가연도": test_year,
            "모델": label,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

        print(
            f"{label}: "
            f"MAE={mae:.2f}, "
            f"RMSE={rmse:.2f}"
        )

        for i in range(len(test)):
            predictions.append({
                "평가연도": test_year,
                "월": int(test.iloc[i]["월"]),
                "지역": test.iloc[i]["지역"],
                "모델": label,
                "실제건수": float(actual[i]),
                "예측건수": float(pred[i])
            })

result_df = pd.DataFrame(results)
prediction_df = pd.DataFrame(predictions)

print("\n===== 전체 성능 비교 =====")
print(result_df.to_string(index=False))

print("\n===== 연도별 MAE 비교 =====")
print(
    result_df.pivot(
        index="평가연도",
        columns="모델",
        values="MAE"
    ).round(2).to_string()
)

# 기존 Random Forest 결과 불러오기
previous_path = DATA_DIR / "walk_forward_results.csv"

if previous_path.exists():
    previous = pd.read_csv(
        previous_path,
        encoding="utf-8-sig"
    )

    previous = previous[
        previous["모델"] == "Random Forest"
    ]

    combined = pd.concat(
        [previous, result_df],
        ignore_index=True
    )

    print("\n===== 기존 RF와 AI 보정 모델 비교 =====")
    print(
        combined.pivot(
            index="평가연도",
            columns="모델",
            values="MAE"
        ).round(2).to_string()
    )

result_df.to_csv(
    DATA_DIR / "residual_model_results.csv",
    index=False,
    encoding="utf-8-sig"
)

prediction_df.to_csv(
    DATA_DIR / "residual_model_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n결과 저장 완료!")
