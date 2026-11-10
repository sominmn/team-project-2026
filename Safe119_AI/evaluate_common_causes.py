
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

df = pd.read_csv(
    DATA_DIR / "monthly_common_cause_counts.csv",
    encoding="utf-8-sig"
)

df["날짜"] = pd.to_datetime(
    dict(year=df["연도"], month=df["월"], day=1)
)

df = df.sort_values(["지역", "날짜"]).copy()

# 지역별 과거 구조활동 이력 생성
group = df.groupby("지역")["구조활동건수"]

df["이전달건수"] = group.shift(1)

df["최근3개월평균"] = (
    df.groupby("지역")["구조활동건수"]
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)

df["작년동월건수"] = group.shift(12)

# 이전 12개월이 있는 2021년부터 학습
df = df.dropna(
    subset=[
        "이전달건수",
        "최근3개월평균",
        "작년동월건수"
    ]
).copy()

base_features = [
    "연도", "월", "지역",
    "이전달건수", "최근3개월평균", "작년동월건수"
]

def create_model(model_type):
    numeric_cols = [
        col for col in base_features if col != "지역"
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
        ("preprocess", preprocessor),
        ("model", estimator)
    ])

results = []

for test_year in range(2022, 2026):
    print(f"\n===== {test_year}년 공통 사고원인 검증 =====")

    train = df[df["연도"] < test_year]
    test = df[df["연도"] == test_year]

    experiments = [
        ("이전 달 기준", None),
        ("Ridge", "Ridge"),
        ("Random Forest", "RF")
    ]

    for label, model_type in experiments:
        if model_type is None:
            prediction = test["이전달건수"].to_numpy()
        else:
            model = create_model(model_type)

            model.fit(
                train[base_features],
                train["구조활동건수"]
            )

            prediction = model.predict(
                test[base_features]
            )

        prediction = np.maximum(prediction, 0)

        mae = mean_absolute_error(
            test["구조활동건수"], prediction
        )

        rmse = np.sqrt(mean_squared_error(
            test["구조활동건수"], prediction
        ))

        results.append({
            "평가연도": test_year,
            "모델": label,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

        print(
            f"{label}: MAE={mae:.2f}, "
            f"RMSE={rmse:.2f}"
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
    ).to_string()
)

output_path = (
    DATA_DIR / "common_cause_model_results.csv"
)

result_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:", output_path)
