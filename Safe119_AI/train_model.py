
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1. AI 학습용 데이터 불러오기
df = pd.read_csv(
    DATA_DIR / "ml_features.csv",
    encoding="utf-8-sig"
)

df["날짜"] = pd.to_datetime(df["날짜"])
df = df.sort_values(["날짜", "지역"])

# 2. 학습/테스트 분리
# 과거 데이터로 학습하고 2025년 데이터로 평가
train = df[df["연도"] <= 2024].copy()
test = df[df["연도"] == 2025].copy()

if train.empty or test.empty:
    raise ValueError("학습 또는 테스트 데이터가 없습니다.")

feature_cols = [
    "연도",
    "월",
    "지역",
    "이전달건수",
    "최근3개월평균",
    "작년동월건수"
]

X_train = train[feature_cols]
y_train = train["구조활동건수"]

X_test = test[feature_cols]
y_test = test["구조활동건수"]

# 3. 지역명 숫자 변환 + AI 모델
preprocessor = ColumnTransformer(
    transformers=[
        (
            "region",
            OneHotEncoder(handle_unknown="ignore"),
            ["지역"]
        )
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "random_forest",
        RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])

# 4. AI 학습
print("AI 모델 학습 중...")
model.fit(X_train, y_train)

# 5. 2025년 데이터 예측
predictions = model.predict(X_test)
predictions = np.maximum(predictions, 0)

# 비교용: 이전 달 건수 그대로 예측
baseline = X_test["이전달건수"].to_numpy()

# 6. 성능 평가
def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    print(f"\n[{name}]")
    print(f"MAE : {mae:,.2f}건")
    print(f"RMSE: {rmse:,.2f}건")

    return mae

baseline_mae = evaluate(
    "기본 예측",
    y_test,
    baseline
)

ai_mae = evaluate(
    "Random Forest AI",
    y_test,
    predictions
)

improvement = (
    (baseline_mae - ai_mae) / baseline_mae * 100
    if baseline_mae != 0 else float("nan")
)

print(f"\nMAE 개선율: {improvement:.2f}%")

# 7. 예측 결과 저장
results = test[
    ["날짜", "지역", "구조활동건수"]
].copy()

results["기본예측"] = baseline
results["AI예측"] = predictions.round(1)
results["절대오차"] = abs(
    results["구조활동건수"] - results["AI예측"]
)

output_path = DATA_DIR / "prediction_results.csv"

results.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

# 8. 학습된 AI 모델 저장
model_path = MODEL_DIR / "random_forest.pkl"
joblib.dump(model, model_path)

print("\n===== 학습 완료 =====")
print("학습 데이터:", len(train), "행")
print("테스트 데이터:", len(test), "행")
print("\n예측 결과 예시:")
print(results.head(10).to_string(index=False))
print("\n모델 저장:", model_path)
print("예측 결과 저장:", output_path)
