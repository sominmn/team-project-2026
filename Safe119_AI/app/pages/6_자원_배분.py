from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from ui import start,plot,csv_button,COLORS,BLUE,RED,GREEN,GRAY

start('구조 자원 배분 모의실험','06 · RESOURCE ALLOCATION','시도별 예측 비중에 따라 가상 자원을 배분하고 실제 공개 기록 비중과 비교합니다.')
OUT=Path(__file__).resolve().parents[2]/'reports/v3'
LABELS={'equal':'균등 배분','baseline':'전월 기준선','rf':'기존 RF','weather_rf':'기상 포함 RF','growth_rf':'변화율 RF'}
@st.cache_data
def load():
    return tuple(pd.read_csv(OUT/f'allocation_{n}.csv') for n in ('detail','monthly','yearly'))
st.caption('Safe119 AI · 후향 평가 · 17개 시도 × 48개월 (2022~2025)')
st.info('실제 인력·장비 배치가 아닌 가상 자원 배분 점수입니다. 평가 기준은 과거 공개 기록의 지역별 비중이며, 예측 시점에는 알 수 없다고 가정합니다.')
try:
    d,m,a=load()
except FileNotFoundError:
    st.error('프로젝트 루트에서 python evaluate_resource_allocation.py 를 실행해 주세요.')
    st.stop()
c1,c2,c3=st.columns(3)
year=c1.selectbox('평가 연도',sorted(d.year.unique(),reverse=True))
month=c2.selectbox('평가 월',list(range(1,13)),index=7)
method=c3.selectbox('배분 방식',list(LABELS),format_func=LABELS.get,index=1)
budget=st.slider('가상 가용 자원 점수',min_value=50,max_value=1000,value=100,step=50)
s=d.loc[d.year.eq(year)&d.month.eq(month)&d.method.eq(method)].copy()
s['배분 점수']=s.allocated_share*budget
s['기록 비중 환산 점수']=s.actual_share*budget
s['배분 차이']=s['배분 점수']-s['기록 비중 환산 점수']
s['미충족 비중 환산 점수']=(-s['배분 차이']).clip(lower=0)
metric=float(s['미충족 비중 환산 점수'].sum())
k1,k2,k3=st.columns(3)
k1.metric('가상 자원 합계',f'{budget:,}점')
k2.metric('공개 기록 건수',f'{s.actual.sum():,}건')
k3.metric('선택 월 배분 불일치',f'{metric:.2f}점',help='배분 비중과 기록 비중의 절대 차이 합의 절반에 가용 점수를 곱한 값입니다.')
st.caption('위 불일치 점수는 선택한 월과 가용 자원 총량 기준입니다. 아래 연도별 비교는 항상 100점 기준입니다.')
st.subheader('시도별 배분 점수와 공개 기록 비중')
show=s.sort_values('기록 비중 환산 점수',ascending=False).rename(columns={'region':'시도','actual':'공개 기록 건수'})
fig=px.bar(show,x='시도',y=['배분 점수','기록 비중 환산 점수'],barmode='group',color_discrete_sequence=[BLUE,'#A9B8C6'],labels={'value':'가상 점수','variable':'구분'})
plot(fig,400)
st.subheader('지역별 배분 차이')
gap=show.sort_values('배분 차이').copy();gap['구분']=gap['배분 차이'].map(lambda x:'기록 비중 대비 과소' if x<0 else '기록 비중 대비 과다 또는 일치')
fig=px.bar(gap,x='배분 차이',y='시도',color='구분',orientation='h',color_discrete_map={'기록 비중 대비 과소':RED,'기록 비중 대비 과다 또는 일치':BLUE},text_auto='.2f');fig.add_vline(x=0,line_dash='dash');plot(fig,570)
st.dataframe(show[['시도','공개 기록 건수','배분 점수','기록 비중 환산 점수','배분 차이']].round(2),use_container_width=True,hide_index=True)
st.subheader('연도별 배분 방식 검증')
annual=a.copy(); annual['방식']=annual.method.map(LABELS)
rank=annual[annual.year==year].sort_values('mean_mismatch_points').copy();base=rank.loc[rank.method=='baseline','mean_mismatch_points'].iloc[0];rank['전월 기준선 대비 개선율(%)']=100*(1-rank.mean_mismatch_points/base);best=rank.iloc[0];chosen=rank[rank.method==method].iloc[0]
c=st.columns(2);c[0].metric('선택 연도 최저 불일치 (사후)',best['방식']);c[0].caption(f"{best.mean_mismatch_points:.3f}점 / 100점");c[1].metric('선택 방식 개선율 (전월 기준선 대비)',f"{chosen['전월 기준선 대비 개선율(%)']:+.2f}%")
rank['순위']=range(1,len(rank)+1)
st.dataframe(rank[['순위','방식','mean_mismatch_points','전월 기준선 대비 개선율(%)']].rename(columns={'mean_mismatch_points':'연평균 불일치 (100점 기준)'}).round(3),hide_index=True,use_container_width=True)
include_equal=st.checkbox('균등 배분도 그래프에 표시',value=False);chart=annual if include_equal else annual[annual.method!='equal'];plot(px.bar(chart,x=chart.year.astype(str),y='mean_mismatch_points',color='방식',barmode='group',color_discrete_map=COLORS,text_auto='.2f',labels={'x':'평가 연도','mean_mismatch_points':'평균 불일치 (100점 기준)'}),370)
st.info('변화율 RF는 기존 RF보다 네 연도 모두 불일치가 작았습니다. 전월 기준선 대비 2022~2024년에는 개선됐으나 2025년에는 더 불리했습니다. 실제 운영 효율을 검증한 결과는 아닙니다.')
wide=annual.pivot(index='year',columns='방식',values='mean_mismatch_points').round(3)
st.dataframe(wide.rename_axis('평가 연도'),use_container_width=True)
st.caption('평가식: 100 × Σ |예측 배분 비중 − 실제 기록 비중| ÷ 2. 매월 동일한 가상 총량을 배분하고 12개월을 동일 가중 평균합니다. 낮을수록 기록 분포와 유사합니다.')
st.warning('사후 비교이며 최적 배분 방식을 사전에 선택한 결과가 아닙니다. 실제 소방 자원 보유량, 출동 시간, 지역별 최소 배치, 긴급도, 동시 출동은 반영하지 않았습니다. 2025년 자료 수록 범위 변화 및 자료 공개 시점 제약도 존재합니다.')
st.download_button('연도별 검증 결과 CSV',a.to_csv(index=False).encode('utf-8-sig'),'allocation_yearly.csv','text/csv')

csv_button(show,'selected_allocation.csv','선택 조건 배분 결과 CSV')
