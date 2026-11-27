"""Retrospective demand-share allocation; no real emergency deployment."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/v3'
KEY = ['year','month','region']
METHODS = {'equal':'균등 배분','baseline':'전월 기준선','rf':'기존 RF','weather_rf':'기상 포함 RF','growth_rf':'변화율 RF'}


def calculate(pred):
    p = pred.loc[pred.target.eq('all') & pred.weather.eq(False), KEY+['actual','baseline','rf','growth_rf']].copy()
    w = pred.loc[pred.target.eq('all') & pred.weather.eq(True), KEY+['actual','rf']].rename(columns={'actual':'weather_actual','rf':'weather_rf'})
    p = p.merge(w,on=KEY,validate='one_to_one',how='outer',indicator=True)
    if not p['_merge'].eq('both').all() or not p.actual.eq(p.weather_actual).all():
        raise ValueError('Weather and non-weather panels do not match')
    if len(p)!=816 or p[KEY].duplicated().any() or p.region.nunique()!=17 or p.groupby(['year','month']).size().ne(17).any() or len(p.groupby(['year','month']))!=48:
        raise ValueError('Expected 48 complete months with 17 regions')
    p['equal']=1.0
    columns=['actual',*METHODS]
    if not np.isfinite(p[columns].to_numpy()).all() or p[columns].lt(0).any().any():
        raise ValueError('Nonfinite or negative counts')
    monthly=p.groupby(['year','month'])[columns].transform('sum')
    if monthly.le(0).any().any():
        raise ValueError('Monthly denominator is zero')
    p['actual_share']=p.actual/monthly.actual
    rows=[]
    for m in METHODS:
        r=p[KEY+['actual','actual_share']].copy()
        r['method']=m
        r['weight']=p[m]
        r['allocated_share']=p[m]/monthly[m]
        r['share_gap']=r.allocated_share-r.actual_share
        rows.append(r)
    detail=pd.concat(rows,ignore_index=True)
    g=detail.groupby(['year','month','method'])
    assert np.allclose(g.allocated_share.sum(),1)
    assert np.allclose(g.actual_share.sum(),1)
    detail['absolute_gap']=detail.share_gap.abs()
    detail['shortfall']=(-detail.share_gap).clip(lower=0)
    month=detail.groupby(['year','month','method'],as_index=False).agg(abs_gap=('absolute_gap','sum'),shortfall=('shortfall','sum'))
    month['mismatch_points']=50*month.abs_gap
    assert np.allclose(month.mismatch_points,100*month.shortfall)
    assert month.mismatch_points.between(0,100).all()
    annual=month.groupby(['year','method'],as_index=False).agg(months=('month','size'),mean_mismatch_points=('mismatch_points','mean'))
    return detail,month[['year','month','method','mismatch_points']],annual

if __name__=='__main__':
    d,m,a=calculate(pd.read_csv(OUT/'predictions.csv'))
    for name,frame in [('allocation_detail',d),('allocation_monthly',m),('allocation_yearly',a)]:
        frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    print(a.pivot(index='year',columns='method',values='mean_mismatch_points').round(3).to_string())
