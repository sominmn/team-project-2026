
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "ml_features.csv", encoding="utf-8-sig")
df["날짜"] = pd.to_datetime(
    dict(year=df["연도"], month=df["월"], day=1)
)
df = df.sort_values(["날짜", "지역"]).reset_index(drop=True)

df["변화량"] = df["구조활동건수"] - df["이전달건수"]

features = [
    "연도", "월", "지역",
    "이전달건수", "최근3개월평균", "작년동월건수"
]
numeric = [
    "연도", "월",
    "이전달건수", "최근3개월평균", "작년동월건수"
]
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

def make_preprocessor(scale=False):
    transformer = StandardScaler() if scale else "passthrough"

    return ColumnTransformer([
        ("region", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False), ["지역"]),
        ("numeric", transformer, numeric)
    ])

def make_model(kind, residual=False):
    if kind == "Ridge":
        estimator = Ridge(alpha=10.0)
        preprocessor = make_preprocessor(scale=True)

    elif kind == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=15,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=42
        )
        preprocessor = make_preprocessor()

    elif kind == "RandomForest":
        estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        preprocessor = make_preprocessor()

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

def adaptive_prediction(test, change):
    test = test.copy()

    for alpha in alphas:
        test[f"pred_{alpha}"] = np.maximum(
            test["이전달건수"].to_numpy() + alpha * change,
            0
        )

    output = np.zeros(len(test))
    selected = {}

    for month in range(1, 13):
        current = test["월"].to_numpy() == month

        if month <= 6:
            alpha = 0.25
        else:
            history = test[
                (test["월"] >= month - 6)
                & (test["월"] < month)
            ]

            errors = {
                a: mean_absolute_error(
                    history["구조활동건수"],
                    history[f"pred_{a}"]
                )
                for a in alphas
            }

            alpha = min(errors, key=errors.get)

        output[current] = test.loc[
            current, f"pred_{alpha}"
        ].to_numpy()

        selected[month] = alpha

    return output, selected

results = []
predictions = []

for year in range(2022, 2026):
    print(f"\n===== {year}년 평가 =====")

    train = df[df["연도"] < year].copy()
    test = df[df["연도"] == year].copy()

    actual = test["구조활동건수"].to_numpy()
    previous = test["이전달건수"].to_numpy()

    candidates = {
        "이전 달 기준": previous
    }

    for kind in ["Ridge", "HistGradientBoosting", "RandomForest"]:
        model = make_model(kind)
        model.fit(train[features], train["구조활동건수"])
        candidates[kind] = np.maximum(
            model.predict(test[features]), 0
        )

    # RF 잔차 예측 모델
    residual_model = make_model("RandomForest", residual=True)
    residual_model.fit(train[features], train["변화량"])

    predicted_change = residual_model.predict(test[features])

    candidates["RF 잔차 고정형 0.25"] = np.maximum(
        previous + 0.25 * predicted_change, 0
    )

    adaptive_pred, selected = adaptive_prediction(
        test, predicted_change
    )
    candidates["RF 잔차 적응형"] = adaptive_pred

    print("월별 선택 alpha:", selected)

    for name, pred in candidates.items():
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))

        results.append({
            "평가연도": year,
            "모델": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

        for i in range(len(test)):
            predictions.append({
                "평가연도": year,
                "월": int(test.iloc[i]["월"]),
                "지역": test.iloc[i]["지역"],
                "모델": name,
                "실제건수": float(actual[i]),
                "예측건수": float(pred[i])
            })

result_df = pd.DataFrame(results)
prediction_df = pd.DataFrame(predictions)

print("\n===== 연도별 MAE =====")
print(
    result_df.pivot(
        index="평가연도",
        columns="모델",
        values="MAE"
    ).round(2).to_string()
)

print("\n===== 연도별 RMSE =====")
print(
    result_df.pivot(
        index="평가연도",
        columns="모델",
        values="RMSE"
    ).round(2).to_string()
)

selection = (
    result_df[result_df["평가연도"].between(2022, 2024)]
    .groupby("모델")[["MAE", "RMSE"]]
    .mean()
    .sort_values("MAE")
)

print("\n===== 2022~2024 평균 성능 순위 =====")
print(selection.round(2).to_string())

result_df.to_csv(
    DATA / "advanced_model_results.csv",
    index=False,
    encoding="utf-8-sig"
)

prediction_df.to_csv(
    DATA / "advanced_model_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n완료! advanced_model_*.csv 저장됨.")
