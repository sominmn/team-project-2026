
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
    preprocess = ColumnTransformer([
        (
            "region",
            OneHotEncoder(handle_unknown="ignore"),
            ["지역"]
        )
    ], remainder="passthrough")

    estimator = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", estimator)
    ])

def evaluate_strategy(test_year, strategy):
    train = df[df["연도"] < test_year].copy()
    test = df[df["연도"] == test_year].copy()

    if strategy == "최근 24개월":
        cutoff = pd.Timestamp(
            year=test_year - 2,
            month=1,
            day=1
        )

        train = train[
            train["날짜"] >= cutoff
        ].copy()

    model = create_model()

    if strategy == "최근 데이터 가중":
        # 학습 종료 시점에서 몇 개월 전 데이터인지
        last_date = train["날짜"].max()

        age_months = (
            (last_date.year - train["날짜"].dt.year) * 12
            + (last_date.month - train["날짜"].dt.month)
        )

        # 12개월 전 데이터는 약 0.5의 가중치
        weights = 0.5 ** (age_months / 12)

        model.fit(
            train[features],
            train["변화량"],
            model__sample_weight=weights
        )
    else:
        model.fit(
            train[features],
            train["변화량"]
        )

    test["AI변화량"] = model.predict(
        test[features]
    )

    for alpha in alphas:
        test[f"예측_{alpha}"] = np.maximum(
            test["이전달건수"]
            + alpha * test["AI변화량"],
            0
        )

    # 최근 6개월 오차를 이용해
    # 매월 보정 강도를 자동으로 선택
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

    actual = test["구조활동건수"]
    prediction = test["적응형예측"]

    mae = mean_absolute_error(actual, prediction)
    rmse = np.sqrt(
        mean_squared_error(actual, prediction)
    )

    return {
        "평가연도": test_year,
        "학습방식": strategy,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    }, test

strategies = [
    "기존 적응형",
    "최근 데이터 가중",
    "최근 24개월"
]

results = []
predictions = []

for year in range(2022, 2026):
    print(f"\n===== {year}년 평가 =====")

    for strategy in strategies:
        result, pred = evaluate_strategy(
            year, strategy
        )

        results.append(result)

        pred["학습방식"] = strategy
        predictions.append(pred)

        print(
            f"{strategy}: "
            f"MAE={result['MAE']:.2f}, "
            f"RMSE={result['RMSE']:.2f}"
        )

result_df = pd.DataFrame(results)
prediction_df = pd.concat(
    predictions,
    ignore_index=True
)

print("\n===== 전체 성능 비교 =====")
print(result_df.to_string(index=False))

print("\n===== 연도별 MAE 비교 =====")
print(
    result_df.pivot(
        index="평가연도",
        columns="학습방식",
        values="MAE"
    ).round(2).to_string()
)

result_df.to_csv(
    DATA_DIR / "weighted_residual_results.csv",
    index=False,
    encoding="utf-8-sig"
)

prediction_df.to_csv(
    DATA_DIR / "weighted_residual_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n결과 저장 완료!")
