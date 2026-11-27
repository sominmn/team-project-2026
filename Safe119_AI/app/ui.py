from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports' / 'v3'
DATA = ROOT / 'data' / 'processed'
NAVY = '#183653'; BLUE = '#3679AF'; ORANGE = '#BC6C24'; GREEN = '#087F72'; RED = '#B84040'; GRAY = '#66778A'
COLORS = {'전월 기준선': GRAY, '기존 RF': BLUE, '기상 포함 RF': ORANGE, '변화율 RF': GREEN, 'Ridge':'#8169A8','HistGB':'#5196A0','최근 2년 RF':'#487E7B','균등 배분':'#BEC9D4','공개 기록':NAVY,'항상 조치':GREEN,'신호 시 조치':ORANGE,'조치 없음':BLUE}
MODELS = {'baseline':'전월 기준선','rf':'기존 RF','ridge':'Ridge','histgb':'HistGB','growth_rf':'변화율 RF','recent_rf':'최근 2년 RF','weather_rf':'기상 포함 RF'}

@st.cache_data
def read_csv(name, report=True):
    return pd.read_csv((REPORTS if report else DATA) / (name + '.csv'), encoding='utf-8-sig')

def start(title, number, subtitle):
    st.set_page_config(page_title=f'Safe119 | {title}', page_icon='🚒', layout='wide')
    st.markdown('''<style>
    .block-container {padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1440px;}
    [data-testid="stMetric"] {background: rgba(120,150,178,.07); border: 1px solid rgba(120,150,178,.25); border-radius: 12px; padding: 15px 16px;}
    [data-testid="stMetricValue"] {font-size: clamp(1.35rem, 2.3vw, 2rem);}
    .safe-hero {background:#183653;color:#fff;padding:26px 30px;border-radius:14px;margin-bottom:14px;border-left:5px solid #D99A52;}
    .safe-kicker {font-size:12px;letter-spacing:1.6px;color:#B8D2E8;font-weight:600;margin-bottom:8px;}
    .safe-title {font-size:clamp(24px,3vw,37px);font-weight:750;line-height:1.3;margin:0 0 8px;}
    .safe-sub {font-size:15px;color:#D8E6F1;line-height:1.65;}
    </style>''', unsafe_allow_html=True)
    st.markdown(f'<div class="safe-hero"><div class="safe-kicker">SAFE119 AI / {number}</div><div class="safe-title">{title}</div><div class="safe-sub">{subtitle}</div></div>',unsafe_allow_html=True)
    st.caption('2020–2025 공개 구조활동 기록 · 2022–2025 후향 평가 · 실시간 출동 예측 서비스 아님')

def plot(fig, height=370):
    fig.update_layout(height=height, margin=dict(l=12,r=15,t=35,b=45),legend=dict(orientation='h',y=-.23,x=0),font=dict(size=13),hoverlabel=dict(namelength=-1))
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        if axis.title.text in ('연도','평가 연도'):
            axis.update(type='category')
    st.plotly_chart(fig, use_container_width=True)

def csv_button(df, name, label='결과 CSV 다운로드'):
    st.download_button(label, df.to_csv(index=False).encode('utf-8-sig'),name,'text/csv')

def predictions():
    return read_csv('predictions')

def model_frame(target='all'):
    p=predictions(); a=p[(p.target==target)&(~p.weather)].copy(); b=p[(p.target==target)&p.weather].copy()
    keys=['year','month','region','actual']
    cols=['baseline','rf','ridge','histgb','recent_rf','growth_rf'] if target=='all' else ['baseline','rf']
    return a[keys+cols].merge(b[keys+['rf']].rename(columns={'rf':'weather_rf'}),on=keys,validate='one_to_one')

def scores_from(frame, cols):
    baseline=(frame.baseline-frame.actual).abs().mean()
    rows=[]
    for m in cols:
        err=frame[m]-frame.actual
        mae=err.abs().mean(); rows.append({'모델':MODELS[m],'MAE':mae,'RMSE':(err.pow(2).mean())**.5,'기준선 대비 개선율(%)':100*(1-mae/baseline) if baseline else float('nan'),'표본수':len(err)})
    return pd.DataFrame(rows).sort_values('MAE')
