
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import jensenshannon

DATA = Path(__file__).resolve().parent / "data" / "processed"

df = pd.read_csv(
    DATA / "audit_cause_counts.csv",
    encoding="utf-8-sig"
)

cause_col = df.columns[0]
years = ["2020", "2021", "2022", "2023", "2024", "2025"]

# Categories with positive counts in every year.
common = df[(df[years] > 0).all(axis=1)].copy()

def js_divergence(data, year_a, year_b):
    p = data[year_a].to_numpy(dtype=float)
    q = data[year_b].to_numpy(dtype=float)

    p = p / p.sum()
    q = q / q.sum()

    distance = jensenshannon(p, q, base=2)
    return float(distance ** 2)

rows = []

for i in range(1, len(years)):
    previous = years[i - 1]
    current = years[i]

    rows.append({
        "previous_year": previous,
        "current_year": current,
        "all_categories_jsd": js_divergence(
            df, previous, current
        ),
        "common_9_jsd": js_divergence(
            common, previous, current
        )
    })

result = pd.DataFrame(rows)

print("\n===== JSD COMPARISON =====")
print(result.round(4).to_string(index=False))

print("\nCommon category count:", len(common))

result.to_csv(
    DATA / "common_vs_all_jsd.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved: common_vs_all_jsd.csv")

