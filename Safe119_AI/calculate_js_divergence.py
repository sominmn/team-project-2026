
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import jensenshannon

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"

source = DATA / "audit_cause_counts.csv"
df = pd.read_csv(source, encoding="utf-8-sig")

category_col = df.columns[0]
years = [str(y) for y in range(2020, 2026)]

# Use a single source and consistent category definitions.
counts = df.set_index(category_col)[years].fillna(0)

results = []

for year in range(2021, 2026):
    previous = str(year - 1)
    current = str(year)

    p = counts[previous].astype(float)
    q = counts[current].astype(float)

    p = p / p.sum()
    q = q / q.sum()

    distance = jensenshannon(p, q, base=2)
    divergence = float(distance ** 2)

    results.append({
        "previous_year": year - 1,
        "current_year": year,
        "js_divergence": divergence,
        "js_distance": float(distance),
    })

result = pd.DataFrame(results)

print("\n===== STANDARDIZED JS RESULTS =====")
print(result.round(4).to_string(index=False))

output = DATA / "js_divergence_standardized.csv"
result.to_csv(output, index=False, encoding="utf-8-sig")

print(f"\nSaved: {output}")
