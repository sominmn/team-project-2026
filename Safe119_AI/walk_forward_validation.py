
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

rescue = pd.read_csv(
    DATA_DIR / "ml_features.csv",
    encoding="utf-8-sig"
)

weather = pd.read_csv(
    DATA_DIR / "monthly_weather_by_region.csv",
    encoding="utf-8-sig"
)

weather_cols = [
    "평균기온", "평균강수량", "평균풍속",
    "최대풍속", "평균습도", "평균구름량"
]

weather = weather.sort_values(
    ["지역", "연도", "월"]
).copy()

# 이전 달 기상 변수 생성
for col in weather_cols:
    weather["이전달_" + col] = (
        weather.groupby("지역")[col].shift(1)
    )

lag_weather = ["이전달_" + col for col in weather_cols]

df = rescue.merge(
    weather[["연도", "월", "지역"] + lag_weather],
    on=["연도", "월", "지역"],
    how="left",
    validate="one_to_one"
)

base_features = [
    "연도", "월", "지역",
    "이전달건수", "최근3개월평균", "작년동월건수"
]

def make_model(model_type, features):
    numeric_cols = [c for c in features if c != "지역"]

    preprocess = ColumnTransformer([
        (
            "region",
            OneHotEncoder(handle_unknown="ignore"),
            ["지역"]
        ),
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_cols
        )
    ])

    if model_type == "Ridge":
        estimator = Ridge(alpha=10.0)
    else:
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

results = []

for test_year in range(2022, 2026):
    print(f"\n===== {test_year}년 검증 중 =====")

    train = df[df["연도"] < test_year].copy()
    test = df[df["연도"] == test_year].copy()

    experiments = [
        ("이전 달 기준", None, base_features),
        ("Ridge", "Ridge", base_features),
        ("Random Forest", "RF", base_features),
        (
            "Random Forest + 날씨",
            "RF",
            base_features + lag_weather
        )
    ]

    for label, model_type, features in experiments:
        if model_type is None:
            predictions = test["이전달건수"].to_numpy()
        else:
            model = make_model(model_type, features)

            model.fit(
                train[features],
                train["구조활동건수"]
            )

            predictions = model.predict(test[features])

        predictions = np.maximum(predictions, 0)

        mae = mean_absolute_error(
            test["구조활동건수"],
            predictions
        )

        rmse = np.sqrt(mean_squared_error(
            test["구조활동건수"],
            predictions
        ))

        results.append({
            "평가연도": test_year,
            "모델": label,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

        print(
            f"{label}: "
            f"MAE={mae:.2f}, RMSE={rmse:.2f}"
        )

result_df = pd.DataFrame(results)

print("\n===== 전체 결과 =====")
print(result_df.to_string(index=False))

print("\n===== 연도별 MAE 비교 =====")
print(
    result_df.pivot(
        index="평가연도",
        columns="모델",
        values="MAE"
    ).round(2).to_string()
)

output_path = DATA_DIR / "walk_forward_results.csv"

result_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:", output_path)
