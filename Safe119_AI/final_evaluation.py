
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

rescue = pd.read_csv(
    DATA_DIR / "ml_features.csv",
    encoding="utf-8-sig"
)

weather = pd.read_csv(
    DATA_DIR / "monthly_weather_by_region.csv",
    encoding="utf-8-sig"
)

weather_columns = [
    "평균기온", "평균강수량", "평균풍속",
    "최대풍속", "평균습도", "평균구름량"
]

weather = weather.sort_values(
    ["지역", "연도", "월"]
).copy()

for col in weather_columns:
    weather["이전달_" + col] = (
        weather.groupby("지역")[col].shift(1)
    )

lag_weather = ["이전달_" + c for c in weather_columns]

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

train = df[df["연도"] <= 2024].copy()
test = df[df["연도"] == 2025].copy()

def create_model(features):
    numeric_features = [
        col for col in features if col != "지역"
    ]

    preprocessor = ColumnTransformer([
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
            numeric_features
        )
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

results = []

# 기본 예측
baseline = test["이전달건수"].values

results.append({
    "모델": "이전 달 기준",
    "MAE": mean_absolute_error(
        test["구조활동건수"], baseline
    ),
    "RMSE": np.sqrt(mean_squared_error(
        test["구조활동건수"], baseline
    ))
})

for label, features in [
    ("Random Forest", base_features),
    ("Random Forest + 날씨", base_features + lag_weather)
]:
    print(f"{label} 학습 중...")

    model = create_model(features)

    model.fit(
        train[features],
        train["구조활동건수"]
    )

    prediction = np.maximum(
        model.predict(test[features]), 0
    )

    results.append({
        "모델": label,
        "MAE": mean_absolute_error(
            test["구조활동건수"], prediction
        ),
        "RMSE": np.sqrt(mean_squared_error(
            test["구조활동건수"], prediction
        ))
    })

    if label == "Random Forest":
        joblib.dump(
            model,
            MODEL_DIR / "safe119_final_model.pkl"
        )

    output = test[
        ["날짜", "지역", "구조활동건수"]
    ].copy()

    output["예측건수"] = prediction

    filename = (
        "final_predictions_weather.csv"
        if "날씨" in label
        else "final_predictions.csv"
    )

    output.to_csv(
        DATA_DIR / filename,
        index=False,
        encoding="utf-8-sig"
    )

result_df = pd.DataFrame(results).sort_values("MAE")

print("\n===== 2025년 최종 평가 =====")
print(result_df.round(2).to_string(index=False))

result_df.to_csv(
    DATA_DIR / "final_model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n최종 평가 완료!")
