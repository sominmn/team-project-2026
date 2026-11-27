import streamlit as st
import pandas as pd
import plotly.express as px
from ui import start,read_csv,plot,csv_button,BLUE,NAVY,RED,ORANGE,GREEN,GRAY,COLORS,REPORTS
start('데이터 정합성 진단','03 · DATA DIAGNOSTICS','공개 기록의 품질, 사고원인 구성, 연도 간 분포 변화와 예측 실패를 분석합니다.')
quality=read_csv('audit_year_quality',False); causes=read_csv('audit_cause_counts',False);j=read_csv('common_vs_all_jsd',False);j['비교 연도']=j.previous_year.astype(str)+' → '+j.current_year.astype(str)
t1,t2,t3,t4=st.tabs(['01 품질 진단','02 분포 변화','03 공통 유형','04 데이터 변화 신호'])
with t1:
 st.subheader('전체 기간 데이터 전처리 요약')
 raw=int(quality['원본행수'].sum()); valid=int(read_csv('monthly_rescue_counts',False)['구조활동건수'].sum()); excluded=raw-valid
 c=st.columns(3);c[0].metric('2020–2025 원본 행',f'{raw:,}행');c[1].metric('집계 제외 행',f'{excluded:,}행');c[2].metric('유효 집계 기록',f'{valid:,}행')
 st.caption('지역 정보가 누락된 기록은 지역·월별 집계에서 제외했습니다.')
 y=st.selectbox('진단 연도',sorted(quality['연도'],reverse=True));r=quality[quality['연도']==y].iloc[0];c=st.columns(4)
 for col,label,key in zip(c,['원본 행 수','사고원인 결측','날짜 파싱 실패','관측 사고원인 종류'],['원본행수','사고원인결측','날짜파싱실패','사고원인종류']):col.metric(label,f'{r[key]:,}')
 st.subheader('연도별 원본 기록 규모');fig=px.bar(quality,x='연도',y='원본행수',text_auto=',.0f',color_discrete_sequence=[BLUE],labels={'원본행수':'원본 행 수'});fig.update_yaxes(tickformat=',.0f');plot(fig)
 st.subheader('연도별 관측 사고원인 종류');fig=px.line(quality,x='연도',y='사고원인종류',markers=True,text='사고원인종류',color_discrete_sequence=[NAVY],labels={'사고원인종류':'관측 사고원인 종류 수'});fig.update_traces(textposition='top center');fig.update_yaxes(range=[0,45],dtick=10);plot(fig,300)
 st.caption('원본 행 수와 유효 지역·월 집계 건수는 지역 결측 등으로 차이가 날 수 있습니다. 명칭 수 변화만으로 공식 분류체계 변경을 확정할 수 없습니다.')
 st.dataframe(quality,hide_index=True,use_container_width=True);csv_button(quality,'year_quality.csv')
with t2:
 c=st.columns(2);c[0].metric('2021→2022 JSD',f"{j.loc[j.current_year==2022,'all_categories_jsd'].iloc[0]:.4f}");c[1].metric('2024→2025 JSD',f"{j.loc[j.current_year==2025,'all_categories_jsd'].iloc[0]:.4f}")
 fig=px.bar(j,x='비교 연도',y='all_categories_jsd',text_auto='.4f',labels={'all_categories_jsd':'JSD'},color_discrete_sequence=[BLUE]);fig.update_traces(marker_color=[RED if x in [2022,2025] else BLUE for x in j.current_year]);fig.update_yaxes(range=[0,1]);plot(fig)
 st.caption('Jensen–Shannon divergence, 로그 밑 2 (0~1). 값이 클수록 구성비 차이가 큽니다. 거리(distance)가 아닌 divergence입니다.')
 st.subheader('기존 RF의 전월 기준선 대비 MAE 배율')
 scores=read_csv('model_scores').query("target=='all' and weather==False and model=='rf'").copy();scores['RF / 전월 기준선']=scores.mae/scores.baseline_mae
 fig=px.bar(scores,x=scores.year.astype(str),y='RF / 전월 기준선',text_auto='.2f',color_discrete_sequence=[BLUE],labels={'x':'평가 연도'});fig.add_hline(y=1,line_dash='dash');plot(fig,320)
 st.info('2022년과 2025년에 큰 분포 변화와 RF 성능 저하가 함께 관찰됐습니다. 두 현상의 인과관계를 입증하지는 않습니다. 연간 JSD는 해당 연도 종료 후 확인한 사후 지표입니다.')
 st.subheader('2024 → 2025 공개 기록의 수록 범위')
 a=causes[(causes['2024']>0)&(causes['2025']==0)].sort_values('2024',ascending=False);c=st.columns(2);c[0].metric('2025년 미관측 명칭',f'{len(a)}개');c[1].metric('해당 명칭의 2024 기록',f"{a['2024'].sum():,}행")
 fig=px.bar(a,x='2024',y='사고원인',orientation='h',color_discrete_sequence=[ORANGE],text_auto=',.0f',labels={'2024':'2024년 기록 행 수'});fig.update_xaxes(tickformat=',.0f');plot(fig,390)
 st.markdown('**벌(집)제거:** 2024년 291,422행 → 2025년 공개 기록 0행. 팀의 조사 기록에는 2025년 공식 생활안전 출동 통계에 벌집 제거 235,804건이 보고된 것으로 정리되어 있습니다. 공식 통계와 공개 기록의 단위·범위가 달라 직접 등치할 수 없습니다.')
 with st.expander('자료 범위 조사 기록 및 출처'): st.markdown((REPORTS/'source_scope.md').read_text(encoding='utf-8'))
with t3:
 st.subheader('2024→2025년 전체 기록과 공통 16유형')
 common=causes[(causes['2024']>0)&(causes['2025']>0)];c=st.columns(2);c[0].metric('전체 원본 행 변화',f"{100*(causes['2025'].sum()/causes['2024'].sum()-1):+.2f}%");c[1].metric(f'두 연도 공통 {len(common)}유형 변화',f"{100*(common['2025'].sum()/common['2024'].sum()-1):+.2f}%")
 q=pd.DataFrame({'연도':['2024','2025'],'전체 기록':[causes['2024'].sum(),causes['2025'].sum()],'공통 16유형':[common['2024'].sum(),common['2025'].sum()]});plot(px.bar(q,x='연도',y=['전체 기록','공통 16유형'],barmode='group',color_discrete_sequence=[NAVY,BLUE],text_auto=',.0f',labels={'value':'기록 행 수','variable':'비교 범위'}).update_yaxes(tickformat=',.0f'))
 st.subheader('6개 연도 공통 9유형');plot(px.bar(j,x='비교 연도',y=['all_categories_jsd','common_9_jsd'],barmode='group',text_auto='.4f',color_discrete_sequence=[NAVY,ORANGE],labels={'value':'JSD','variable':'집계 범위'}).for_each_trace(lambda t:t.update(name={'all_categories_jsd':'전체 유형','common_9_jsd':'공통 9유형'}.get(t.name,t.name))))
 st.dataframe(j[['비교 연도','all_categories_jsd','common_9_jsd']].rename(columns={'all_categories_jsd':'전체 유형 JSD','common_9_jsd':'공통 9유형 JSD'}).round(4),hide_index=True,use_container_width=True)
 st.caption('공통 9유형은 2020~2025년 전체에서 관측된 명칭으로 사후 선별했습니다. 분포 차이는 선별 유형 내부의 구성비 차이이며 전체 수록 범위의 일치를 의미하지 않습니다.')
 st.subheader('공통 9유형 재평가: 기준선 대비 RF 성능')
 s=read_csv('model_scores').query("target=='common' and weather==False and model=='rf'");plot(px.bar(s,x=s.year.astype(str),y='improvement_pct',text_auto='.2f',color_discrete_sequence=[RED],labels={'x':'평가 연도','improvement_pct':'전월 기준선 대비 RF 개선율 (%)'}),320)
 st.dataframe(s[['year','baseline_mae','mae','improvement_pct']].rename(columns={'year':'평가 연도','baseline_mae':'전월 기준선 MAE','mae':'RF MAE','improvement_pct':'기준선 대비 개선율(%)'}).round(2),hide_index=True,use_container_width=True)
 st.info('공통 명칭만 남겨도 RF가 기준선을 넘지 못했습니다. 공통 명칭의 정의 일치는 확인되지 않았으며, 분류 차이만으로 성능 문제를 설명하거나 보정 효과를 주장할 수 없습니다.')
 with st.expander('사고원인 명칭 대응 후보 (정의 미확인)'):st.dataframe(read_csv('cause_name_crosswalk'),hide_index=True,use_container_width=True)
with t4:
 st.subheader('최종 v3 평가: 지역별 연간 경보율과 기준선 초과 MAE');s=read_csv('signal_correlations');plot(px.bar(s,x=s.year.astype(str),y='spearman',text_auto='.3f',color_discrete_sequence=[BLUE],labels={'x':'연도','spearman':'Spearman 상관계수'}),320)
 st.info('일관된 양의 상관관계가 확인되지 않았습니다. 지역별 연간 경보율 분석으로 개별 월의 오차 탐지 성능 전체를 판단할 수 없으며, 실제 재난 경보나 자원 배치 판단에 사용하지 않습니다.')
 v=read_csv('signal_region_year');year=st.selectbox('v3 상관 분석 연도',[2025,2024,2023,2022]);vv=v[v.year==year];plot(px.scatter(vv,x='alert_rate',y='excess_mae',hover_name='region',labels={'alert_rate':'경보율','excess_mae':'RF 기준선 초과 MAE (건)'},color_discrete_sequence=[BLUE]).update_xaxes(tickformat='.0%'),340)
 p=read_csv('predictions').query("target=='all' and weather==False");st.dataframe(p[['year','recent_threshold','seasonal_threshold']].drop_duplicates().assign(recent_threshold=lambda x:x.recent_threshold.map(lambda z:f'{z:.2%}'),seasonal_threshold=lambda x:x.seasonal_threshold.map(lambda z:f'{z:.2%}')).rename(columns={'year':'평가 연도','recent_threshold':'최근 급변 임계값','seasonal_threshold':'계절 이탈 임계값'}),hide_index=True,use_container_width=True);st.caption('2022년은 기존 30%·40%, 이후에는 이전 평가 기간의 90백분위수로 산정. 후향 평가입니다.')
 st.subheader('기존 탐색 실험 (고정 30%·40%, 최종 v3와 별도)');st.caption('최근 변화율: 직전월과 전전월의 절대 차이를 전전월 건수로 나눕니다. 계절 변화율: 직전월과 전년도 동일 월의 절대 차이를 전년도 동일 월 건수로 나눕니다. 두 신호 중 하나면 WATCH, 둘 다면 HIGH이며 재난 위험도가 아닙니다.');r=read_csv('risk_signals_monthly',False);c=st.columns(3);yy=c[0].selectbox('탐색 연도',sorted(r['연도'].unique(),reverse=True));mm=c[1].selectbox('탐색 월',list(range(1,13)),index=7);rg=c[2].selectbox('탐색 지역',sorted(r['지역'].unique()));z=r[(r['연도']==yy)&(r['월']==mm)];u=z[z['지역']==rg].iloc[0];c=st.columns(3);c[0].metric('변화 신호',u['위험판정']);c[1].metric('최근 변화율',f"{u['최근변화율']:.1%}" if pd.notna(u['최근변화율']) else '자료 부족');c[2].metric('계절 변화율',f"{u['계절변화율']:.1%}" if pd.notna(u['계절변화율']) else '자료 부족')
 counts=z['위험판정'].value_counts().reindex(['NORMAL','WATCH','HIGH','INSUFFICIENT'],fill_value=0).rename_axis('신호').reset_index(name='지역 수');plot(px.bar(counts,x='신호',y='지역 수',color='신호',color_discrete_map={'NORMAL':BLUE,'WATCH':ORANGE,'HIGH':RED,'INSUFFICIENT':GRAY},text_auto=True),280)
 with st.expander('고정 임계값·RF 잔차 모델의 기존 오차 및 전환 실험'):
  st.caption('v3와 모델·임계값·평가 조건이 다릅니다. 사후 탐색 결과이며 최종 비교표에 합치지 않습니다.')
  for file in ['risk_signal_performance','risk_signal_comparison','model_switching_yearly','model_switching_by_risk']:
   st.markdown('**'+{'risk_signal_performance':'수준별 오차','risk_signal_comparison':'신호 유무별 오차','model_switching_yearly':'연도별 전환 성능','model_switching_by_risk':'수준별 전환 성능'}[file]+'**');st.dataframe(read_csv(file,False),hide_index=True,use_container_width=True)
  old=read_csv('model_switching_yearly',False)
  fig=px.bar(old,x=old.year.astype(str),y='switch_vs_ai_pct',text_auto='.2f',labels={'x':'연도','switch_vs_ai_pct':'항상 잔차 RF 대비 전환 개선율 (%)'});fig.update_traces(marker_color=[GREEN if x>0 else RED for x in old.switch_vs_ai_pct]);fig.add_hline(y=0,line_dash='dash');plot(fig,300)
  st.caption('고정 임계값 전환은 2025년에만 항상 잔차 RF보다 개선되었으며, 2022~2024년에는 악화되었습니다.')
  csv_button(z,'exploratory_signals.csv')
