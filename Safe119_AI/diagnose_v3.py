from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent
data = root / "data/processed"
out = root / "reports/v3"
out.mkdir(parents=True, exist_ok=True)

causes = pd.read_csv(data / "audit_cause_counts.csv")
common = set(pd.read_csv(data / "common_causes_2020_2025.csv")["사고원인"])
causes["공통9유형에포함"] = causes["사고원인"].isin(common)
causes["정의확인"] = "미확인"
causes["판정근거"] = "연도별 공개 집계의 명칭과 관측 건수"
causes.to_csv(out / "cause_name_crosswalk.csv", index=False, encoding="utf-8-sig")

panel = pd.read_csv(data / "monthly_rescue_counts.csv")
index = pd.MultiIndex.from_product([range(2020, 2026), range(1, 13), sorted(panel["지역"].unique())], names=["연도", "월", "지역"])
coverage = panel.groupby(["연도", "월", "지역"]).size().reindex(index, fill_value=0).rename("집계행수").reset_index()
coverage.to_csv(out / "aggregate_coverage.csv", index=False, encoding="utf-8-sig")
