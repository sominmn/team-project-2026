
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
    "연도",
    "월",
    "지역",
    "이전달건수",
    "최근3개월평균",
    "작년동월건수"
]

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
limits = [0.1, 0.2, 0.3, 0.5]

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

    return Pipeline([
        ("preprocess", preprocess),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

def adaptive_predict(test, limit=None):
    test = test.copy()

    # 보정량 제한 기준은 예측 시점에
    # 이미 알려진 최근 3개월 평균만 사용
    change = test["AI변화량"].to_numpy()

    if limit is not None:
        max_change = (
            test["최근3개월평균"].to_numpy()
            * limit
        )

        change = np.clip(
            change,
            -max_change,
            max_change
        )

    for alpha in alphas:
        test[f"예측_{alpha}"] = np.maximum(
            test["이전달건수"].to_numpy()
            + alpha * change,
            0
        )

    test["적응형예측"] = np.nan
    test["선택alpha"] = np.nan

    for month in range(1, 13):
        current_mask = test["월"] == month

        if month <= 6:
            selected_alpha = 0.25
        else:
            history = test[
                (test["월"] >= month - 6)
                & (test["월"] < month)
            ]

            errors = {
                alpha: mean_absolute_error(
                    history["구조활동건수"],
                    history[f"예측_{alpha}"]
                )
                for alpha in alphas
            }

            selected_alpha = min(
                errors,
                key=errors.get
            )

        test.loc[
            current_mask, "적응형예측"
        ] = test.loc[
            current_mask, f"예측_{selected_alpha}"
        ]

        test.loc[
            current_mask, "선택alpha"
        ] = selected_alpha

    return test

results = []
predictions = []

for year in range(2022, 2026):
    print(f"\n===== {year}년 평가 =====")

    train = df[df["연도"] < year].copy()
    test = df[df["연도"] == year].copy()

    model = create_model()

    model.fit(
        train[features],
        train["변화량"]
    )

    test["AI변화량"] = model.predict(
        test[features]
    )

    experiments = {}

    # 기준 모델
    experiments["이전 달 기준"] = (
        test["이전달건수"].to_numpy()
    )

    # 보정량 제한 없음
    unlimited = adaptive_predict(test)

    experiments["고정형 0.25"] = (
        unlimited["예측_0.25"].to_numpy()
    )

    experiments["적응형 제한 없음"] = (
        unlimited["적응형예측"].to_numpy()
    )

    # 보정량 제한 실험
    for limit in limits:
        clipped = adaptive_predict(
            test,
            limit=limit
        )

        experiments[f"고정형 제한 {limit}"] = (
            clipped["예측_0.25"].to_numpy()
        )

        experiments[f"적응형 제한 {limit}"] = (
            clipped["적응형예측"].to_numpy()
        )

    actual = test["구조활동건수"].to_numpy()

    for label, pred in experiments.items():
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(
            mean_squared_error(actual, pred)
        )

        results.append({
            "평가연도": year,
            "모델": label,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

        for i in range(len(test)):
            predictions.append({
                "평가연도": year,
                "월": int(test.iloc[i]["월"]),
                "지역": test.iloc[i]["지역"],
                "모델": label,
                "실제건수": float(actual[i]),
                "예측건수": float(pred[i])
            })

result_df = pd.DataFrame(results)
prediction_df = pd.DataFrame(predictions)

print("\n===== 연도별 MAE 비교 =====")
print(
    result_df.pivot(
        index="평가연도",
        columns="모델",
        values="MAE"
    ).round(2).to_string()
)

print("\n===== 연도별 RMSE 비교 =====")
print(
    result_df.pivot(
        index="평가연도",
        columns="모델",
        values="RMSE"
    ).round(2).to_string()
)

# 2022~2024년 평균으로 후보 비교
# 2025년은 모델 선택에 사용하지 않음
selection = (
    result_df[
        result_df["평가연도"].between(2022, 2024)
    ]
    .groupby("모델")[["MAE", "RMSE"]]
    .mean()
    .sort_values("MAE")
)

print("\n===== 2022~2024년 평균 성능 순위 =====")
print(selection.round(2).to_string())

result_df.to_csv(
    DATA_DIR / "clipped_residual_results.csv",
    index=False,
    encoding="utf-8-sig"
)

prediction_df.to_csv(
    DATA_DIR / "clipped_residual_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n결과 저장 완료!")
