
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

# 기존 구조활동 학습 데이터
rescue = pd.read_csv(
    DATA_DIR / "ml_features.csv",
    encoding="utf-8-sig"
)

# 날씨 데이터
weather = pd.read_csv(
    DATA_DIR / "monthly_weather_by_region.csv",
    encoding="utf-8-sig"
)

weather_columns = [
    "평균기온",
    "평균강수량",
    "평균풍속",
    "최대풍속",
    "평균습도",
    "평균구름량"
]

# 지역별 시간순 정렬
weather = weather.sort_values(
    ["지역", "연도", "월"]
).copy()

# 이전 달 날씨 생성
for col in weather_columns:
    weather["이전달_" + col] = (
        weather.groupby("지역")[col].shift(1)
    )

lag_weather_columns = [
    "이전달_" + col for col in weather_columns
]

# 구조활동 데이터와 결합
df = rescue.merge(
    weather[
        ["연도", "월", "지역"] + lag_weather_columns
    ],
    on=["연도", "월", "지역"],
    how="left",
    validate="one_to_one"
)

base_features = [
    "연도",
    "월",
    "지역",
    "이전달건수",
    "최근3개월평균",
    "작년동월건수"
]

weather_features = base_features + lag_weather_columns

# 학습/검증/최종 테스트 구분
train = df[df["연도"] <= 2023].copy()
valid = df[df["연도"] == 2024].copy()
test = df[df["연도"] == 2025].copy()

def make_model(model_name, use_weather):
    features = (
        weather_features if use_weather else base_features
    )

    numeric_features = [
        col for col in features if col != "지역"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        (
            "region",
            OneHotEncoder(handle_unknown="ignore"),
            ["지역"]
        ),
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    ])

    if model_name == "Ridge":
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
        ("preprocessor", preprocessor),
        ("model", estimator)
    ]), features

def evaluate(actual, predicted):
    predicted = np.maximum(predicted, 0)
    return (
        mean_absolute_error(actual, predicted),
        np.sqrt(mean_squared_error(actual, predicted))
    )

results = []

print("\n===== 2024년 검증 데이터 성능 비교 =====")

# 이전 달 건수를 사용하는 기준 모델
baseline_mae, baseline_rmse = evaluate(
    valid["구조활동건수"],
    valid["이전달건수"]
)

results.append({
    "모델": "이전 달 기준",
    "MAE": round(baseline_mae, 2),
    "RMSE": round(baseline_rmse, 2)
})

for model_name in ["Ridge", "RandomForest"]:
    for use_weather in [False, True]:
        label = (
            f"{model_name} + 날씨"
            if use_weather else model_name
        )

        model, features = make_model(
            model_name, use_weather
        )

        model.fit(
            train[features],
            train["구조활동건수"]
        )

        predictions = model.predict(valid[features])

        mae, rmse = evaluate(
            valid["구조활동건수"],
            predictions
        )

        results.append({
            "모델": label,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

result_df = pd.DataFrame(results).sort_values("MAE")

print(result_df.to_string(index=False))

output_path = DATA_DIR / "weather_model_comparison.csv"

result_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n검증 결과 저장:", output_path)
print("2025년 최종 테스트는 아직 실행하지 않았습니다.")
