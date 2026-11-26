"""Reproduce the v3 retrospective evaluation from tracked monthly data."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/processed"
OUT = ROOT / "reports/v3"
KEY = ["year", "month", "region"]
RENAME = dict(zip(["연도", "월", "지역", "구조활동건수"], KEY + ["actual"]))
YEARS = range(2022, 2026)
ACTIONS = ["growth_rf", "recent_rf", "baseline"]


def read(name):
    return pd.read_csv(DATA / f"{name}.csv").rename(columns=RENAME)


def panel(name, weather):
    source = read(name)[KEY + ["actual"]].set_index(KEY)
    index = pd.MultiIndex.from_product(
        [range(2020, 2026), range(1, 13), sorted(source.index.unique("region"))],
        names=KEY,
    )
    frame = source.reindex(index).reset_index()
    frame = frame.merge(weather, on=KEY, how="left", validate="one_to_one")
    grouped = frame.groupby("region")
    for lag in [1, 2, 3, 12, 13]:
        frame[f"lag{lag}"] = grouped["actual"].shift(lag)
    frame["mean3"] = frame[["lag1", "lag2", "lag3"]].mean(axis=1, skipna=False)
    frame["growth"] = frame["actual"].div(frame["lag1"].replace(0, np.nan)) - 1
    frame["recent_change"] = frame["lag1"].sub(frame["lag2"]).abs().div(frame["lag2"].replace(0, np.nan))
    frame["seasonal_change"] = frame["lag1"].sub(frame["lag13"]).abs().div(frame["lag13"].replace(0, np.nan))
    return frame


def rf():
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=119, n_jobs=2)


def scores(frame, groups):
    result = frame.groupby(groups).agg(n=("error", "size"), mae=("error", "mean"), baseline_mae=("baseline_error", "mean"))
    result["improvement_pct"] = 100 * (1 - result["mae"].div(result["baseline_mae"].replace(0, np.nan)))
    return result.reset_index()


def save(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def main(*, include_shap=True):
    OUT.mkdir(parents=True, exist_ok=True)
    weather = read("monthly_weather_by_region")
    weather_cols = sorted(set(weather.columns) - set(KEY))
    weather["date"] = pd.to_datetime(dict(year=weather.year, month=weather.month, day=1)) + pd.offsets.MonthBegin(1)
    weather["year"] = weather.date.dt.year
    weather["month"] = weather.date.dt.month
    features = ["month", "lag1", "lag2", "mean3", "lag12"]
    blocks = []
    for target, name in [("all", "monthly_rescue_counts"), ("common", "monthly_common_cause_counts")]:
        frame = panel(name, weather[KEY + weather_cols])
        for year in YEARS:
            training = frame.loc[frame.year.lt(year)].dropna(subset=features + ["actual"])
            testing = frame.loc[frame.year.eq(year)].dropna(subset=features + ["actual"]).copy()
            # Prior years only. Unobserved signals remain unclassified.
            calibration = frame.loc[frame.year.between(2022, min(2024, year - 1))]
            changes = ["recent_change", "seasonal_change"]
            thresholds = calibration[changes].quantile(.9)
            thresholds = thresholds.fillna(pd.Series(dict(zip(changes, [.30, .40]))))
            valid = testing[changes].notna().all(axis=1)
            testing["signal"] = testing[changes].ge(thresholds).any(axis=1).where(valid)
            testing["recent_threshold"], testing["seasonal_threshold"] = thresholds
            for with_weather in [False, True]:
                cols = features + (weather_cols if with_weather else [])
                train_x = training[cols]
                test_x = testing[cols]
                estimator = rf()
                fitted = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), estimator).fit(train_x, training.actual)
                block = testing[KEY + ["actual", "signal", "recent_threshold", "seasonal_threshold"]].copy()
                block["target"], block["weather"] = target, with_weather
                block["baseline"] = testing.lag1
                block["rf"] = fitted.predict(test_x)
                if target == "all" and not with_weather:
                    block["ridge"] = make_pipeline(StandardScaler(), Ridge()).fit(train_x, training.actual).predict(test_x)
                    block["histgb"] = HistGradientBoostingRegressor(random_state=119).fit(train_x, training.actual).predict(test_x)
                    recent = training.year.ge(year - 2)
                    block["recent_rf"] = rf().fit(train_x.loc[recent], training.loc[recent, "actual"]).predict(test_x)
                    growth = training.growth.notna()
                    block["growth_rf"] = (testing.lag1 * (1 + rf().fit(train_x.loc[growth], training.loc[growth, "growth"]).predict(test_x))).clip(lower=0)
                    if year == 2025 and include_shap:
                        explain(estimator, testing[features], testing.region)
                blocks.append(block)
    predictions = pd.concat(blocks, ignore_index=True)
    save(predictions, "predictions")
    key = KEY + ["actual", "signal", "target", "weather"]
    model_names = ["baseline", "rf", "ridge", "histgb"]
    long = predictions.melt(id_vars=key, value_vars=model_names, var_name="model", value_name="prediction").dropna(subset=["prediction"])
    long["signed_error"] = long.prediction - long.actual
    long["error"] = long.signed_error.abs()
    long = long.merge(
        long.loc[long.model.eq("baseline"), key[:-3] + ["target", "weather", "error"]].rename(columns={"error": "baseline_error"}),
        on=KEY + ["actual", "target", "weather"], validate="many_to_one",
    )
    save(scores(long, ["year", "target", "weather", "model"]), "model_scores")
    save(scores(long.query("target == 'all' and not weather"), ["year", "region", "model"]), "regional_scores")
    save(long.query("target == 'all' and not weather and year == 2025"), "errors_2025")
    evaluate_actions(predictions)


def evaluate_actions(predictions):
    subset = predictions.query("target == 'all' and not weather").copy()
    subset["baseline_error"] = subset.baseline.sub(subset.actual).abs()
    rf_error = subset.rf.sub(subset.actual).abs()
    alert = subset.signal.eq(True)
    rows = []
    labels = []
    for action in ACTIONS:
        action_error = subset[action].sub(subset.actual).abs()
        for policy, error in [
            ("none", rf_error),
            ("always", action_error),
            ("signal_only", action_error.where(alert, rf_error)),
        ]:
            rows.append(subset[KEY + ["baseline_error"]].assign(action=action, policy=policy, error=error))
        labeled = subset[KEY + ["signal"]].assign(action=action, rf_error=rf_error, action_error=action_error)
        labeled["outcome"] = np.select(
            [subset.signal.isna(), alert & action_error.lt(rf_error), alert, action_error.lt(rf_error)],
            ["unknown", "helpful_alert", "false_alarm", "missed_benefit"],
            default="no_alert_no_benefit",
        )
        labels.append(labeled)
    save(scores(pd.concat(rows), ["year", "action", "policy"]), "action_scores")
    outcomes = pd.concat(labels)
    save(outcomes, "alert_detail")
    save(outcomes.groupby(["year", "action", "outcome"]).size().rename("count").reset_index(), "alert_counts")
    annual = subset.assign(alert=alert.astype(int), usable=subset.signal.notna(), rf_error=rf_error)
    annual = annual.groupby(["year", "region"]).agg(months=("rf_error", "size"), usable=("usable", "sum"), alerts=("alert", "sum"), mae=("rf_error", "mean"), baseline_mae=("baseline_error", "mean")).reset_index()
    annual["excess_mae"] = annual.mae - annual.baseline_mae
    annual["alert_rate"] = annual.alerts.div(annual.usable.replace(0, np.nan))
    save(annual, "signal_region_year")
    correlations = annual.groupby("year")[["alert_rate", "excess_mae"]].apply(
        lambda group: group.alert_rate.corr(group.excess_mae, method="spearman")
    )
    save(correlations.rename("spearman").reset_index(), "signal_correlations")


def explain(estimator, features, regions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    values = shap.TreeExplainer(estimator)(features)
    for plot in (shap.plots.bar, shap.plots.beeswarm):
        plot(values, show=False)
        plt.savefig(OUT / f"shap_{plot.__name__}.png", dpi=160, bbox_inches="tight")
        plt.close()
    save(pd.DataFrame(values.values, columns=features.columns).assign(region=regions.to_numpy()), "shap_2025")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-shap", action="store_true", help="Generate evaluation tables only")
    main(include_shap=not parser.parse_args().skip_shap)
