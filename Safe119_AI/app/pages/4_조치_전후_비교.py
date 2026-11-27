import pandas as pd
import streamlit as st
import plotly.express as px
from ui import start,read_csv,plot,csv_button,COLORS,GRAY,BLUE,GREEN,ORANGE
start('조치 전후 예측 비교','04 · INTERVENTION','변화율 예측, 최근 2년 학습, 전월 기준선 전환의 효과를 동일한 v3 평가 표본에서 검증합니다.')
ACTIONS={'growth_rf':'변화율 RF','recent_rf':'최근 2년 RF','baseline':'전월 기준선 전환'};POLICIES={'none':'조치 없음','always':'항상 조치','signal_only':'신호 시 조치'}
p=read_csv('predictions').query("target=='all' and weather==False");a=read_csv('action_scores');c=st.columns(3);year=c[0].selectbox('평가 연도',[2025,2024,2023,2022]);region=c[1].selectbox('시도',sorted(p.region.unique()));action=c[2].selectbox('조치 방식',list(ACTIONS),format_func=ACTIONS.get)
st.caption('조치 없음 = 기존 RF · 항상 조치 = 선택한 방식 전 표본 적용 · 신호 시 조치 = 경보가 있는 표본만 선택 방식 적용')
st.write({'growth_rf':'변화율 RF: 직전월 대비 변화율을 예측하고 건수로 환산합니다.','recent_rf':'최근 2년 RF: 평가 연도 직전 2개 연도 자료로 학습합니다.','baseline':'전월 기준선 전환: 직전월 관측 건수를 예측값으로 사용합니다.'}[action])
st.subheader('전국 전략별 MAE')
score=a[(a.year==year)&(a.action==action)].copy();score['전략']=score.policy.map(POLICIES);c=st.columns(3)
for col,policy in zip(c,['none','always','signal_only']):
 r=score[score.policy==policy].iloc[0];col.metric(POLICIES[policy]+' MAE',f'{r.mae:,.2f}건');col.caption(f'전월 기준선 대비 {r.improvement_pct:+.2f}%')
st.caption('전국 204개 시도·월 표본의 평균 절대오차입니다. 개선율 = 100 × (1 − 전략 MAE / 전월 기준선 MAE). 양수는 개선, 음수는 악화입니다. 아래 지역별 그래프는 선택 시도에 해당합니다.')
st.subheader('지역별 실제 기록과 조치 전후 예측');s=p[(p.year==year)&(p.region==region)].sort_values('month');q=pd.DataFrame({'월':s.month,'공개 기록':s.actual,'조치 없음':s.rf,'항상 조치':s[action],'신호 시 조치':s[action].where(s.signal.eq(True),s.rf)})
fig=px.line(q,x='월',y=list(q.columns[1:]),markers=True,color_discrete_map=COLORS,labels={'value':'건수','variable':'전략'});fig.update_xaxes(dtick=1);plot(fig,390)
st.subheader('연도별 전략 효과 (전국)');v=a[a.action==action].copy();v['전략']=v.policy.map(POLICIES);fig=px.bar(v,x=v.year.astype(str),y='improvement_pct',color='전략',barmode='group',color_discrete_map=COLORS,text_auto='.1f',labels={'x':'평가 연도','improvement_pct':'전월 기준선 대비 개선율 (%)'});fig.add_hline(y=0,line_dash='dash');plot(fig,370)
st.info('변화율 RF를 항상 적용하면 2022~2024년 기준선 대비 MAE가 줄었지만, 2025년에는 개선되지 않았습니다. 경보가 있을 때만 적용하는 전략은 변화율 RF의 항상 적용보다 네 연도 모두 불리했습니다. 이 결과는 사후 비교이며 사전 전략 선택을 검증한 것은 아닙니다.')
st.dataframe(v[['year','전략','n','mae','baseline_mae','improvement_pct']].rename(columns={'year':'연도','n':'표본수','mae':'MAE','baseline_mae':'기준선 MAE','improvement_pct':'기준선 대비 개선율(%)'}).round(2),hide_index=True,use_container_width=True)
st.subheader('지역별 데이터 변화 신호와 조치 결과');alerts=read_csv('alert_detail').query('year==@year and region==@region and action==@action').copy();alerts['판정']=alerts.outcome.map({'unknown':'판정 유보','helpful_alert':'경보 및 조치 이득','false_alarm':'경보 있으나 이득 없음','missed_benefit':'조치 이득 있으나 경보 없음','no_alert_no_benefit':'경보 및 조치 이득 없음'});st.dataframe(alerts[['month','signal','rf_error','action_error','판정']].rename(columns={'month':'월','signal':'경보','rf_error':'기존 RF 절대오차','action_error':'선택 방식 적용 시 절대오차'}).round(2),hide_index=True,use_container_width=True)
st.caption('조치 이득은 실제 기록을 사용해 사후 판정합니다. 선택 방식 적용 시 절대오차는 경보와 무관하게 해당 방식을 적용했을 때의 오차입니다. 신호는 데이터 변화 지표이며 실제 재난 경보가 아닙니다.')
csv_button(v,'v3_action_scores.csv')
