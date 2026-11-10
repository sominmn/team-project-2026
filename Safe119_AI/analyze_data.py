
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

# 한글 폰트 설정 (Windows)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 1. 연도별 구조활동 건수
yearly = df.groupby("연도")["구조활동건수"].sum()

print("=== 연도별 구조활동 건수 ===")
print(yearly.to_string())

plt.figure(figsize=(9, 5))
yearly.plot(kind="bar")
plt.title("연도별 119 구조활동 건수")
plt.xlabel("연도")
plt.ylabel("구조활동 건수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / "yearly_rescue.png", dpi=200)
plt.close()

# 2. 지역별 구조활동 건수
regional = (
    df.groupby("지역")["구조활동건수"]
    .sum()
    .sort_values(ascending=False)
)

print("\n=== 지역별 구조활동 건수 ===")
print(regional.to_string())

plt.figure(figsize=(11, 6))
regional.plot(kind="bar")
plt.title("지역별 119 구조활동 건수 (2020~2025)")
plt.xlabel("지역")
plt.ylabel("구조활동 건수")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(output_dir / "regional_rescue.png", dpi=200)
plt.close()

# 3. 월별 구조활동 건수
monthly = df.groupby("월")["구조활동건수"].sum()

print("\n=== 월별 구조활동 건수 ===")
print(monthly.to_string())

plt.figure(figsize=(9, 5))
monthly.plot(kind="line", marker="o")
plt.title("월별 119 구조활동 건수 (2020~2025 합계)")
plt.xlabel("월")
plt.ylabel("구조활동 건수")
plt.xticks(range(1, 13))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "monthly_rescue.png", dpi=200)
plt.close()

print("\n분석 완료!")
print("그래프 저장 위치:", output_dir)