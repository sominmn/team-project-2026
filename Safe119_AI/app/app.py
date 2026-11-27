import streamlit as st
import plotly.express as px
from ui import start,read_csv,plot,MODELS,BLUE,NAVY,GREEN,csv_button
start('Safe119 AI','01 · RESEARCH OVERVIEW','기상데이터 연계 119 출동 수요 예측과 설명가능한 AI를 통한 구조 자원 효율화 시스템')
st.markdown('**연구 목적**  |  공개 구조활동 기록의 예측 가능성과 한계를 검증하고, 설명가능한 AI와 가상 자원 배분을 통해 의사결정 지원 가능성을 탐색합니다.')
d=read_csv('monthly_rescue_counts',False); m=read_csv('model_scores'); a=read_csv('action_scores'); alloc=read_csv('allocation_yearly')
rank=alloc.pivot(index='year',columns='method',values='mean_mismatch_points'); wins=int((rank.growth_rf<rank.baseline).sum())
r=m.query("year==2025 and target=='all' and weather==False and model=='rf'").iloc[0]
g=a.query("action=='growth_rf' and policy=='always'").sort_values('improvement_pct').iloc[-1]
c=st.columns(3); c[0].metric('분석 기간','2020–2025'); c[1].metric('시도',str(d['지역'].nunique())+'개'); c[2].metric('지역·월 집계',f'{len(d):,}개')
st.subheader('핵심 연구 결과')
c=st.columns(3);c[0].metric('2025년 RF MAE / 전월 기준선',f'{r.mae/r.baseline_mae:.2f}배');c[1].metric(f'{int(g.year)}년 변화율 RF 개선 (전월 대비)',f'{g.improvement_pct:.2f}%');c[2].metric('변화율 RF 배분 개선 (전월 대비)',f'{wins} / {len(rank)}개 연도',help='전월 기준선 대비 가상 배분 불일치가 작았던 연도 수')
st.markdown('기상 연계만으로는 RF 성능이 개선되지 않았습니다. 연도 간 기록 구성 변화를 진단하고, 예측 방식 변경과 SHAP 설명, 배분 모의실험을 통해 결과를 비교했습니다.')
st.subheader('연구 흐름')
for title,desc in [('01 데이터 진단','원자료 품질·사고원인 구성·JSD'),('02 수요 예측','전월 기준선·AI·기상 포함 비교'),('03 조치 검증','변화율 예측·경보 기반 전환'),('04 XAI','기상 및 과거 기록의 SHAP 기여도'),('05 자원 배분','예측 비중에 따른 가상 배분과 사후 검증')]:
    st.markdown(f'**{title}** — {desc}')
st.divider();st.subheader('연도·지역별 공개 기록 현황')
c1,c2=st.columns(2);year=c1.selectbox('분석 연도',sorted(d['연도'].unique(),reverse=True));region=c2.selectbox('지역',['전국']+sorted(d['지역'].unique()))
s=d[d['연도']==year];s=s if region=='전국' else s[s['지역']==region];q=s.groupby('월',as_index=False)['구조활동건수'].sum(); c=st.columns(3);c[0].metric('연간 공개 기록',f"{q['구조활동건수'].sum():,}건");c[1].metric('월평균',f"{q['구조활동건수'].mean():,.0f}건");c[2].metric('최다 기록 월',f"{q.loc[q['구조활동건수'].idxmax(),'월']}월")
fig=px.line(q,x='월',y='구조활동건수',markers=True,color_discrete_sequence=[BLUE],labels={'구조활동건수':'공개 기록 건수'});fig.update_xaxes(dtick=1);plot(fig)
with st.expander('월별 상세 데이터'): st.dataframe(q,hide_index=True,use_container_width=True);csv_button(q,'monthly_records.csv')
st.info('자료 범위: 소방청 공개 구조활동 기록 및 기상청 ASOS. 실제 전체 출동이나 자원 수요와 동일하지 않으며, 전월 집계의 예측 시점 접근을 가정한 후향 연구입니다. 실제 인력·장비 배치 효과는 검증하지 않았습니다.')
