
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

input_path = DATA_DIR / "monthly_rescue_counts.csv"
output_path = DATA_DIR / "ml_features.csv"

df = pd.read_csv(input_path, encoding="utf-8-sig")

# 날짜 생성 및 정렬
df["날짜"] = pd.to_datetime(
    dict(year=df["연도"], month=df["월"], day=1)
)

df = df.sort_values(["지역", "날짜"]).copy()

# 지역별로 이전 달 구조활동 건수 계산
grouped = df.groupby("지역")["구조활동건수"]

df["이전달건수"] = grouped.shift(1)

# 이전 3개월의 평균 구조활동 건수
df["최근3개월평균"] = grouped.transform(
    lambda x: x.shift(1).rolling(3).mean()
)

# 작년 같은 달 구조활동 건수
df["작년동월건수"] = grouped.shift(12)

# 예측 시점에 필요한 과거 데이터가 없는 행 제외
df = df.dropna(
    subset=[
        "이전달건수",
        "최근3개월평균",
        "작년동월건수"
    ]
).copy()

# AI 학습에 사용할 컬럼
features = df[
    [
        "날짜",
        "연도",
        "월",
        "지역",
        "이전달건수",
        "최근3개월평균",
        "작년동월건수",
        "구조활동건수"
    ]
]

features.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("===== AI 학습용 데이터 생성 완료 =====")
print("전체 행 수:", len(features))
print("\n처음 10행:")
print(features.head(10).to_string(index=False))
print("\n저장 위치:", output_path)
