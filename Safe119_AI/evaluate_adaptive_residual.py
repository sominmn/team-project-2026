
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

df["날짜"] = pd.to_datetime(
    dict(year=df["연도"], month=df["월"], day=1)
)

df = df.sort_values(
    ["날짜", "지역"]
).reset_index(drop=True)

df["변화량"] = (
    df["구조활동건수"] - df["이전달건수"]
)

features = [
    "연도", "월", "지역",
    "이전달건수", "최근3개월평균", "작년동월건수"
]

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

def create_model():
    preprocess = ColumnTransformer(
        [
            (
                "region",
                OneHotEncoder(handle_unknown="ignore"),
                ["지역"]
            )
        ],
        remainder="passthrough"
    )

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

all_predictions = []

# 매년 이전 연도 데이터만 학습
for test_year in range(2022, 2026):
    print(f"\n===== {test_year}년 평가 =====")

    train = df[df["연도"] < test_year]
    test = df[df["연도"] == test_year]

    model = create_model()

    model.fit(
        train[features],
        train["변화량"]
    )

    test = test.copy()

    test["AI변화량"] = model.predict(
        test[features]
    )

    for alpha in alphas:
        test[f"예측_{alpha}"] = np.maximum(
            test["이전달건수"] + alpha * test["AI변화량"],
            0
        )

    # 평가 연도 이전의 오차를 사용하지 않으므로
    # 처음 6개월은 고정형 0.25 적용
    selected_predictions = []
    selected_alphas = []

    for month in range(1, 13):
        current = test[test["월"] == month]

        if month <= 6:
            selected_alpha = 0.25
        else:
            history = test[
                (test["월"] >= month - 6)
                & (test["월"] < month)
            ]

            errors = {}

            for alpha in alphas:
                errors[alpha] = mean_absolute_error(
                    history["구조활동건수"],
                    history[f"예측_{alpha}"]
                )

            selected_alpha = min(
                errors,
                key=errors.get
            )

        selected_alphas.append(
            (month, selected_alpha)
        )

        selected_predictions.extend(
            zip(
                current.index,
                current[f"예측_{selected_alpha}"]
            )
        )

    selected_series = pd.Series(
        dict(selected_predictions)
    )

    test["적응형예측"] = selected_series

    print("월별 선택 α:", selected_alphas)

    all_predictions.append(test)

result = pd.concat(
    all_predictions,
    ignore_index=True
)

experiments = {
    "이전 달 기준": "예측_0.0",
    "고정 AI 보정 0.25": "예측_0.25",
    "적응형 AI 보정": "적응형예측"
}

rows = []

for year in range(2022, 2026):
    subset = result[result["연도"] == year]

    for label, column in experiments.items():
        actual = subset["구조활동건수"]
        pred = subset[column]

        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(
            mean_squared_error(actual, pred)
        )

        rows.append({
            "평가연도": year,
            "모델": label,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

summary = pd.DataFrame(rows)

print("\n===== 최종 성능 비교 =====")
print(summary.to_string(index=False))

summary.to_csv(
    DATA_DIR / "adaptive_residual_results.csv",
    index=False,
    encoding="utf-8-sig"
)

result.to_csv(
    DATA_DIR / "adaptive_residual_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료!")
