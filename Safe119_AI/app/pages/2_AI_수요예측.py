import streamlit as st
import pandas as pd
import plotly.express as px
from ui import start, model_frame,scores_from,MODELS,COLORS,plot,csv_button,read_csv
start('AI 구조활동 수요 예측','02 · FORECASTING','동일 표본에서 전월 기준선, AI 모델과 직전월 기상 연계 효과를 비교합니다.')
c=st.columns(3);year=c[0].selectbox('평가 연도',[2025,2024,2023,2022]);target=c[1].radio('집계 대상',['all','common'],format_func=lambda x:'전체 공개 기록' if x=='all' else '공통 9유형',horizontal=True)
f=model_frame(target);region=c[2].selectbox('평가 지역',['전국']+sorted(f.region.unique()));s=f[f.year==year];s=s if region=='전국' else s[s.region==region];cols=[k for k in MODELS if k in s.columns];score=scores_from(s,cols)
base=score.loc[score['모델']=='전월 기준선','MAE'].iloc[0];best=score.iloc[0];rf=score.loc[score['모델']=='기존 RF','MAE'].iloc[0];wrf=score.loc[score['모델']=='기상 포함 RF','MAE'].iloc[0]
c=st.columns(3);c[0].metric('전월 기준선 MAE',f'{base:,.2f}건');c[1].metric('최저 MAE (사후 비교)',f"{best.MAE:,.2f}건");c[1].caption(best['모델']);c[2].metric('기상 추가 RF 오차 변화',f'{100*(wrf/rf-1):+.2f}%' if rf else '계산 불가')
st.caption(f'평가 표본: {len(s)}개 시도·월 관측치. 전국 MAE는 전국 월합계 오차가 아니라 시도·월 오차의 평균입니다.')
metric=st.radio('평가 지표',['MAE','RMSE'],horizontal=True);v=score.sort_values(metric,ascending=False)
fig=px.bar(v,x=metric,y='모델',color='모델',orientation='h',color_discrete_map=COLORS,text_auto='.2f');fig.update_layout(showlegend=False);plot(fig,390)
st.caption('MAE: 평균 절대오차 · RMSE: 제곱평균제곱근오차 · 단위: 건. 개선율은 MAE 기준이며 낮은 오차가 우수합니다.')
st.dataframe(score.round(2),hide_index=True,use_container_width=True)
st.subheader('월별 실제 기록과 예측 비교');selected=st.multiselect('표시 모델',cols,default=['baseline','rf','growth_rf'] if target=='all' else ['baseline','rf'],format_func=MODELS.get)
q=s.groupby('month',as_index=False)[['actual']+selected].sum().rename(columns={'actual':'공개 기록',**MODELS});fig=px.line(q,x='month',y=[x for x in q.columns if x!='month'],markers=True,color_discrete_map=COLORS,labels={'month':'월','value':'건수','variable':'구분'});fig.update_xaxes(dtick=1);plot(fig);st.caption('전국 선택 시 그래프는 시도별 예측을 월별로 합산한 값입니다.' if region=='전국' else f'{region}의 월별 공개 기록과 예측입니다.')
st.subheader('직전월 기상데이터 연계 효과 (전국)');rows=[]
for y,group in f.groupby('year'):
 for k in ['rf','weather_rf']: rows.append({'연도':str(y),'모델':MODELS[k],'MAE':(group[k]-group.actual).abs().mean()})
w=pd.DataFrame(rows);plot(px.bar(w,x='연도',y='MAE',color='모델',barmode='group',color_discrete_map=COLORS,text_auto='.1f'))
if target=='all':st.info('전체 기록에서 직전월 기상 포함 RF는 평가한 4개 연도 모두 기상 미포함 RF보다 MAE가 증가했습니다. 기상 자체가 무용하다는 뜻은 아닙니다.')
with st.expander('평가 방법 및 상세 예측 데이터'):
 st.write('평가 연도 이전 자료로 학습한 v3 후향 결과. 기준선 대비 개선율 = 100 × (1 − 모델 MAE / 기준선 MAE). 모델 선택 결과는 사후 비교이며 사전 최적 모델 선택을 입증하지 않습니다.');st.dataframe(s.rename(columns=MODELS).round(2),hide_index=True,use_container_width=True);csv_button(s,'v3_selected_predictions.csv')
