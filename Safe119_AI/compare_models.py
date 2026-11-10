
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(
    DATA_DIR / "ml_features.csv",
    encoding="utf-8-sig"
)

# 시간 순서에 따라 학습/테스트 분리
train = df[df["연도"] <= 2024].copy()
test = df[df["연도"] == 2025].copy()

features = [
    "연도",
    "월",
    "지역",
    "이전달건수",
    "최근3개월평균",
    "작년동월건수"
]

X_train = train[features]
y_train = train["구조활동건수"]

X_test = test[features]
y_test = test["구조활동건수"]

# 모델별 전처리
linear_preprocessor = ColumnTransformer([
    ("region", OneHotEncoder(handle_unknown="ignore"), ["지역"]),
    ("numeric", StandardScaler(), [
        "연도", "월", "이전달건수",
        "최근3개월평균", "작년동월건수"
    ])
])

forest_preprocessor = ColumnTransformer([
    ("region", OneHotEncoder(handle_unknown="ignore"), ["지역"])
], remainder="passthrough")

models = {
    "Ridge Regression": Pipeline([
        ("preprocessor", linear_preprocessor),
        ("model", Ridge(alpha=10.0))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", forest_preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = []

def evaluate(name, predictions):
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append({
        "모델": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })

# 기본 예측
evaluate("Previous Month Baseline", X_test["이전달건수"].values)

# 머신러닝 모델
for name, model in models.items():
    print(f"{name} 학습 중...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    evaluate(name, predictions)

result_df = pd.DataFrame(results).sort_values("MAE")

print("\n===== 모델 성능 비교 =====")
print(result_df.to_string(index=False))

result_df.to_csv(
    DATA_DIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n모델 비교 완료!")
