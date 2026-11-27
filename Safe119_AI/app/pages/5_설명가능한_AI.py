from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from ui import start, plot, BLUE, ORANGE, NAVY, csv_button

start('설명가능한 AI (XAI)','05 · EXPLAINABILITY','Random Forest의 예측 근거를 TreeSHAP으로 설명하고, 직전월 기상 변수의 기여도를 비교합니다.')
OUT=Path(__file__).resolve().parents[2]/'reports/v3'
NAMES={'month':'예측 대상 월','lag1':'직전월 기록','lag2':'전전월 기록','mean3':'직전 3개월 평균','lag12':'전년도 동월 기록'}
@st.cache_data
def load():
    return tuple(pd.read_csv(OUT/f'xai_extended_{n}.csv') for n in ('detail','importance','validation'))
st.caption('Safe119 AI · 2025년 전체 공개 기록 예측 · Random Forest / TreeSHAP')
try:
    d,g,v=load()
except FileNotFoundError:
    st.error('프로젝트 루트에서 python evaluate_xai_extension.py 를 실행해 주세요.')
    st.stop()
st.info('SHAP은 모델 예측에 기여한 변수의 크기와 방향을 보여줍니다. 실제 구조활동의 발생 원인이나 예측 오류의 인과관계를 증명하지는 않습니다.')
c1,c2=st.columns(2)
for c,with_weather in [(c1,False),(c2,True)]:
    m=v.loc[v.weather.eq(with_weather)].iloc[0]
    c.metric('기상 포함 RF MAE' if with_weather else '기상 미포함 RF MAE',f'{m.mae:,.2f}건')
    if bool(m.reference_matches): c.caption('저장된 v3 예측값과 재학습 예측값 일치 (수치 허용오차 내)')
    else: c.warning('v3 저장 예측값과 재학습 결과가 달라 별도 실험으로 해석해야 합니다.')
st.subheader('전체 변수 중요도')
with_weather=st.radio('설명할 RF 모델',[False,True],format_func=lambda x:'직전월 기상 포함' if x else '기상 미포함',horizontal=True,index=1)
importance=g.loc[g.weather.eq(with_weather)].copy()
importance['변수']=importance.feature.map(lambda f:NAMES.get(f,'직전월 '+f))
importance=importance.sort_values('mean_abs_shap')
plot(px.bar(importance,x='mean_abs_shap',y='변수',orientation='h',color='feature_group',labels={'mean_abs_shap':'평균 절대 SHAP 값 (건)','feature_group':'변수군'},color_discrete_map={'과거 구조활동·계절':BLUE,'기상·관측정보':ORANGE},text_auto='.2f').update_yaxes(categoryorder='array',categoryarray=importance['변수'].tolist()),520 if with_weather else 300)
st.caption('204개 시도·월 표본에서 평균 절대 기여도를 계산했습니다. 기상 및 관측정보의 영향 크기는 모델 내부 설명이며, 기상정보의 실질적 예측 효용은 별도 MAE 비교로 판단합니다.')
st.subheader('지역·월별 예측 근거')
st.caption('선택 모델: '+('직전월 기상 포함 RF' if with_weather else '기상 미포함 RF'))
c1,c2=st.columns(2)
region=c1.selectbox('시도',sorted(d.region.unique()))
month=c2.selectbox('월',list(range(1,13)),index=7)
s=d.loc[d.weather.eq(with_weather)&d.region.eq(region)&d.month.eq(month)].copy()
s['변수']=s.feature.map(lambda f:NAMES.get(f,'직전월 '+f))
s['기여 방향']=s.shap_value.map(lambda x:'예측 증가' if x>=0 else '예측 감소')
s['절대 기여도']=s.shap_value.abs()
s=s.sort_values('절대 기여도')
p=float(s.prediction.iloc[0]); actual=float(s.actual.iloc[0]); base=float(s.base_value.iloc[0]); contribution=float(s.shap_value.sum())
a,b,c=st.columns(3)
a.metric('모델 기준값',f'{base:,.1f}건')
b.metric('최종 예측값',f'{p:,.1f}건')
c.metric('실제 공개 기록',f'{actual:,.0f}건',delta=f'{abs(p-actual):,.1f}건 '+('과소 예측' if p<actual else '과대 예측' if p>actual else '일치'),delta_color='off')
plot(px.bar(s,x='shap_value',y='변수',color='기여 방향',orientation='h',color_discrete_map={'예측 증가':BLUE,'예측 감소':ORANGE},labels={'shap_value':'SHAP 기여도 (건)'},).update_yaxes(categoryorder='array',categoryarray=s['변수'].tolist()),530 if with_weather else 300)
st.caption('개별 기여도와 Waterfall 색상: 파란색은 예측값 증가, 주황색은 감소입니다. 전체 변수 중요도 그래프의 변수군 색상과 의미가 다릅니다.')
st.write(f'기준값 {base:,.2f} + SHAP 기여도 합 {contribution:,.2f} = 예측값 {p:,.2f}건 (표시값 반올림)')
top=s.sort_values('절대 기여도',ascending=False).iloc[0]
st.markdown(f"**주요 예측 요인:** {top['변수']}의 기여도는 {top.shap_value:+,.2f}건으로, 선택한 표본에서 절대 기여도가 가장 큽니다.")
with st.expander('SHAP Waterfall — 기준값에서 예측값까지',expanded=True):
    w=s.sort_values('절대 기여도',ascending=False)
    fig=go.Figure(go.Waterfall(orientation='h',measure=['absolute']+['relative']*len(w)+['total'],y=['모델 기준값']+w['변수'].tolist()+['최종 예측값'],x=[base]+w.shap_value.tolist()+[0],increasing={'marker':{'color':BLUE}},decreasing={'marker':{'color':ORANGE}},totals={'marker':{'color':NAVY}},connector={'line':{'color':'#8C9BA8'}}))
    fig.update_yaxes(autorange='reversed');fig.update_xaxes(title='예측 기록 건수');plot(fig,540 if with_weather else 340)
units={'month':'월','lag1':'건','lag2':'건','lag12':'건','mean3':'건','평균기온':'℃','평균강수량':'mm','평균풍속':'m/s','최대풍속':'m/s','평균습도':'%','평균구름량':'할(0~10)','관측지점수':'개','최대적설':'cm','최대3시간신적설':'cm'}
s['입력 단위']=s.feature.map(units).fillna('원자료 단위')
s['입력값 대체 여부']=s.imputed.map({True:'학습 중앙값 대체',False:'원래 입력값'})
st.dataframe(s.sort_values('절대 기여도',ascending=False)[['변수','feature_value','입력 단위','입력값 대체 여부','shap_value']].rename(columns={'feature_value':'모델 입력값','shap_value':'SHAP 기여도 (건)'}).round(3),hide_index=True,use_container_width=True)
st.warning('2025년은 후향 탐색 사례입니다. 직전월 집계 및 기상정보를 예측 시점에 이용할 수 있다고 가정합니다. SHAP 설명은 기본 RF에 대한 것으로, 변화율 RF의 설명으로 해석하지 않습니다.')
st.download_button('변수 중요도 CSV',g.to_csv(index=False).encode('utf-8-sig'),'xai_extended_importance.csv','text/csv')

csv_button(s,'selected_shap_detail.csv','선택 지역·월 SHAP CSV')
