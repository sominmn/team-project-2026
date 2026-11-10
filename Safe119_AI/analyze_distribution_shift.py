
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import jensenshannon

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

# 앞에서 생성한 연도별 사고원인 집계
df = pd.read_csv(
    DATA_DIR / "all_year_cause_comparison.csv",
    encoding="utf-8-sig",
    index_col=0
)

years = list(range(2020, 2026))

# 모든 연도의 사고원인 목록을 같은 기준으로 정렬
counts = df[[str(y) for y in years]].fillna(0)

# 연도별 전체 건수로 나누어 비율 계산
distributions = counts.div(
    counts.sum(axis=0),
    axis=1
)

results = []

for year in range(2021, 2026):
    previous = str(year - 1)
    current = str(year)

    distance = jensenshannon(
        distributions[previous].values,
        distributions[current].values,
        base=2
    )

    results.append({
        "비교연도": f"{year-1}→{year}",
        "JSD": round(float(distance), 4)
    })

result_df = pd.DataFrame(results)

print("\n===== 연도별 사고원인 분포 차이 =====")
print(result_df.to_string(index=False))

print("\n===== 분포 차이가 가장 큰 구간 =====")
print(
    result_df.sort_values(
        "JSD", ascending=False
    ).head(3).to_string(index=False)
)

output_path = (
    DATA_DIR / "distribution_shift_results.csv"
)

result_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:", output_path)

