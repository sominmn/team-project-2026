
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

data_path = (
    BASE_DIR / "data" / "processed"
    / "monthly_rescue_counts.csv"
)

output_dir = BASE_DIR / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path, encoding="utf-8-sig")

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 비교할 지역
regions = [
    "서울특별시",
    "경기도",
    "부산광역시",
    "강원도"
]

# 지역별 월평균 구조활동 건수 계산
filtered = df[df["지역"].isin(regions)]

monthly_avg = (
    filtered.groupby(["지역", "월"])["구조활동건수"]
    .mean()
    .reset_index()
)

# 지역별 월간 패턴 시각화
plt.figure(figsize=(11, 6))

for region in regions:
    region_data = monthly_avg[
        monthly_avg["지역"] == region
    ]

    plt.plot(
        region_data["월"],
        region_data["구조활동건수"],
        marker="o",
        label=region
    )

plt.title("지역별 월평균 119 구조활동 건수")
plt.xlabel("월")
plt.ylabel("월평균 구조활동 건수")
plt.xticks(range(1, 13))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "regional_monthly_pattern.png",
    dpi=200
)

plt.show()

print("지역별 월간 패턴 분석 완료!")
