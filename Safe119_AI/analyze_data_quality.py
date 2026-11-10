
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"

OUT.mkdir(parents=True, exist_ok=True)

# 1. 기존 감사 결과 읽기
cause = pd.read_csv(
    OUT / "audit_cause_counts.csv",
    encoding="utf-8-sig"
)

detail = pd.read_csv(
    OUT / "audit_detail_changes.csv",
    encoding="utf-8-sig"
)

# 2. 2024~2025년 사고원인 분포 차이
a = cause["2024"].to_numpy(dtype=float)
b = cause["2025"].to_numpy(dtype=float)

p = a / a.sum()
q = b / b.sum()

def kl_divergence(x, y):
    mask = x > 0
    return np.sum(
        x[mask] * np.log2(x[mask] / y[mask])
    )

m = (p + q) / 2

jsd = 0.5 * (
    kl_divergence(p, m)
    + kl_divergence(q, m)
)

print("\n===== 2024~2025 사고원인 분포 차이 =====")
print(f"Jensen-Shannon Divergence: {jsd:.4f}")
print("0에 가까울수록 분포가 유사함")
print("1에 가까울수록 분포가 크게 다름")

# 3. 2025년 결측 원본 분석
path = RAW / "소방청_구조활동현황_20251231.csv"

if not path.exists():
    raise FileNotFoundError(path)

rows = []
total = 0
missing_count = 0

for chunk in pd.read_csv(
    path,
    encoding="cp949",
    dtype=str,
    chunksize=50000
):
    total += len(chunk)

    cause_missing = (
        chunk["사고원인"].isna()
        | chunk["사고원인"].str.strip().eq("")
    )

    detail_missing = (
        chunk["사고원인코드명_사고종별"].isna()
        | chunk["사고원인코드명_사고종별"].str.strip().eq("")
    )

    missing = chunk[cause_missing | detail_missing].copy()

    missing_count += len(missing)

    if len(missing) == 0:
        continue

    dates = pd.to_datetime(
        missing["신고년월일"],
        errors="coerce",
        format="mixed"
    )

    missing["월"] = dates.dt.month

    rows.append(
        missing[["월", "발생장소_시"]]
    )

if rows:
    missing_df = pd.concat(rows, ignore_index=True)
else:
    missing_df = pd.DataFrame(
        columns=["월", "발생장소_시"]
    )

monthly_missing = (
    missing_df.groupby("월")
    .size()
    .reindex(range(1, 13), fill_value=0)
    .reset_index(name="결측건수")
)

regional_missing = (
    missing_df.groupby("발생장소_시")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="결측건수")
)

monthly_missing.to_csv(
    OUT / "quality_missing_by_month.csv",
    index=False,
    encoding="utf-8-sig"
)

regional_missing.to_csv(
    OUT / "quality_missing_by_region.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 2025년 결측 분석 =====")
print(f"전체 원본 기록: {total:,}")
print(f"사고원인 또는 세부유형 결측: {missing_count:,}")
print(f"결측 비율: {missing_count / total * 100:.3f}%")

print("\n월별 결측 건수")
print(monthly_missing.to_string(index=False))

print("\n지역별 결측 건수 상위 10개")
print(regional_missing.head(10).to_string(index=False))

# 4. 공통 사고원인의 세부유형 변화
detail["2024에만있는세부유형"] = (
    detail["2024에만있는세부유형"].fillna("")
)
detail["2025에만있는세부유형"] = (
    detail["2025에만있는세부유형"].fillna("")
)

detail["세부유형변화여부"] = (
    (detail["2024에만있는세부유형"] != "")
    | (detail["2025에만있는세부유형"] != "")
)

changed = detail[
    detail["세부유형변화여부"]
].copy()

print("\n===== 세부유형 변화 =====")
print(f"공통 사고원인: {len(detail)}개")
print(f"세부유형 변화 확인: {len(changed)}개")

print(
    changed[
        [
            "사고원인",
            "2024세부유형수",
            "2025세부유형수",
            "공통세부유형수"
        ]
    ].to_string(index=False)
)

# 5. 연구용 요약 파일
summary = pd.DataFrame([
    {
        "항목": "2024~2025 사고원인 JSD",
        "값": round(jsd, 4)
    },
    {
        "항목": "2025년 사고유형 결측 건수",
        "값": missing_count
    },
    {
        "항목": "2025년 사고유형 결측 비율(%)",
        "값": round(missing_count / total * 100, 3)
    },
    {
        "항목": "공통 사고원인 중 세부유형 변화",
        "값": len(changed)
    }
])

summary.to_csv(
    OUT / "quality_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 최종 요약 =====")
print(summary.to_string(index=False))

print("\n분석 완료! quality_*.csv 파일 저장됨.")
