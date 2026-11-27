"""Explain 2025 v3 RFs with and without lagged weather; do not overwrite v3 scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from evaluate_v3 import read, panel, rf, KEY

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/v3'
FEATURES=['month','lag1','lag2','mean3','lag12']


def main():
    weather=read('monthly_weather_by_region')
    weather_cols=sorted(set(weather.columns)-set(KEY))
    date=pd.to_datetime(dict(year=weather.year,month=weather.month,day=1))+pd.offsets.MonthBegin(1)
    weather['year'],weather['month']=date.dt.year,date.dt.month
    frame=panel('monthly_rescue_counts',weather[KEY+weather_cols])
    train=frame.loc[frame.year.lt(2025)].dropna(subset=FEATURES+['actual'])
    test=frame.loc[frame.year.eq(2025)].dropna(subset=FEATURES+['actual']).copy()
    ref=pd.read_csv(OUT/'predictions.csv')
    rows=[]; meta=[]
    for with_weather in (False,True):
        cols=FEATURES+(weather_cols if with_weather else [])
        fit=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),rf()).fit(train[cols],train.actual)
        x=fit.steps[0][1].transform(test[cols])
        model=fit.steps[1][1]
        prediction=fit.predict(test[cols])
        shap_result=shap.TreeExplainer(model)(x)
        values=np.asarray(shap_result.values)
        bases=np.asarray(shap_result.base_values).reshape(-1)
        if bases.size==1: bases=np.repeat(bases,len(test))
        residual=np.abs(bases+values.sum(axis=1)-prediction)
        if not np.allclose(bases+values.sum(axis=1),prediction,rtol=1e-7,atol=1e-5):
            raise ValueError('SHAP additivity check failed')
        compare=test[KEY].assign(retrained=prediction).merge(ref.loc[ref.target.eq('all')&ref.weather.eq(with_weather)&ref.year.eq(2025),KEY+['rf']],on=KEY,how='outer',validate='one_to_one')
        if len(compare)!=204 or compare[['retrained','rf']].isna().any().any(): raise ValueError('Reference keys mismatch')
        delta=float((compare.retrained-compare.rf).abs().max())
        meta.append(dict(weather=with_weather,train_rows=len(train),test_rows=len(test),mae=float(np.abs(prediction-test.actual.to_numpy()).mean()),reference_max_abs_diff=delta,reference_matches=bool(np.allclose(compare.retrained,compare.rf,atol=1e-6,rtol=1e-9)),shap_additivity_max_error=float(residual.max())))
        for j,col in enumerate(cols):
            block=test[KEY+['actual']].copy()
            block['weather']=with_weather
            block['feature']=col
            block['feature_group']='기상·관측정보' if col in weather_cols else '과거 구조활동·계절'
            block['feature_value']=x[:,j]
            block['imputed']=test[col].isna().to_numpy()
            block['shap_value']=values[:,j]
            block['base_value']=bases
            block['prediction']=prediction
            rows.append(block)
    d=pd.concat(rows,ignore_index=True)
    summary=d.assign(abs_shap=d.shap_value.abs()).groupby(['weather','feature','feature_group'],as_index=False).agg(mean_abs_shap=('abs_shap','mean'),n=('abs_shap','size'))
    d.to_csv(OUT/'xai_extended_detail.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(OUT/'xai_extended_importance.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(meta).to_csv(OUT/'xai_extended_validation.csv',index=False,encoding='utf-8-sig')
    (OUT/'xai_extended_versions.json').write_text(json.dumps({'sklearn':sklearn.__version__,'shap':shap.__version__,'pandas':pd.__version__,'numpy':np.__version__},indent=2),encoding='utf-8')
    print(pd.DataFrame(meta).to_string(index=False))
    print(summary.sort_values(['weather','mean_abs_shap'],ascending=[True,False]).to_string(index=False))

if __name__=='__main__': main()
