########국가온실가스종합관리시스템 내 업체별, 지역별 온실가스 및 에너지 사용 현황

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# csv파일 읽기
datas = pd.read_csv('c://python/2019업종별.csv')
datas2 = pd.read_csv('C://python/2019지역별.csv', encoding='cp949')

################# 필요 설정 요소들 #################################
# 글씨체 설정
fp = fm.FontProperties(fname='c://python/KakaoBold.ttf')
fp1 = fm.FontProperties(fname='c://python/KakaoRegular.ttf')
mpl.rc('font', family='Malgun Gothic')
mpl.rc('axes', unicode_minus=False)

# 차트 색상 설정
colors_list = ['#333BD5', '#FF6600', '#777777', '#ECBA3C', '#0099FF',
               '#99CCFF', '#C0C0C0', '#33CC33', '#0033CC', '#FFFF00', ]

colors_list1 = ['#333BD5', '#777777', '#FF6600', '#0099FF', '#C0C0C0',
                '#99CCFF', '#ECBA3C', '#33CC33', '#FFFF00', '#FFCC66', ]

colors_list2 = ['#A9444A', '#F98660', '#F45B96', '#EFC45E', '#8CB559', '#477F5F', '#348DA5', '#36598F', '#983890']

# 파이차트 출력시 공간 간격 및 차트 너비 폭 지정
wedgeprops = {'width': 0.8, 'edgecolor': 'w', 'linewidth': 5}

#################업종별 데이터셋 사용##########################
######1. 업종별 온실가스 배출량
# 필요한 컬럼만 가지고 데이터프레임 만들기
df = pd.DataFrame(datas, columns=['지정업종', '온실가스 배출량(tCO2-eq)'])

# 합계 도출을 위해 object(문자형)데이터를 float(실수)로 변환
# 숫자에 자동으로 삽입된 ',' 없애기
df['온실가스 배출량(tCO2-eq)'] = df['온실가스 배출량(tCO2-eq)'].str.replace(',', '').astype(float)

# 지정업종별로 그룹을 나누어 합계 도출
a = df.groupby('지정업종').sum()

# 온실가스 배출량을 내림차순 기준으로 정렬
b = a.sort_values('온실가스 배출량(tCO2-eq)', ascending=False)

# 상위 10개 행 추출
c = b.head(10)

# 수평 막대 그래프로 표현하기 위해 y축의 길이 정해주기
y = np.arange(10)

# 단위를 십만으로 축약하여 숫자 추출
c_values = [int(i / 100000) for i in list(c['온실가스 배출량(tCO2-eq)'])]

# matplotlib를 활용한 시각화
plt.figure(figsize=(20, 6))
plt.title('업종별 온실가스 배출량', loc='center', fontsize=15, fontweight='bold', fontproperties=fp)
plt.title('(단위 : 십만(tCO2-eq))', loc='right', fontsize=12, fontproperties=fp1)
plt.barh(y, c_values, color=colors_list, alpha=0.6)
plt.yticks(y, c.index, fontproperties=fp1)
plt.xlabel('온실가스 배출량', fontproperties=fp1, fontsize=15)
plt.ylabel('배출량 상위 10개 업종', fontproperties=fp1, fontsize=15)

# 그래프 끝 마디에 c_values의 각 값을 텍스트로 출력 추가
# plt.text는 첫 3개의 인자는 각각 텍스트가 출력될 (x축의 위치, y축의 위치, 출력할 텍스트)이다.
for i in range(len(y)):
    plt.text(c_values[i], y[i], str(c_values[i]),
             fontsize=11,
             color='black',
             horizontalalignment='left',
             verticalalignment='center'
             )
plt.show()

# 2. 업종별 에너지 사용량
# 필요한 컬럼만 가지고 데이터프레임 만들기
df1 = pd.DataFrame(datas, columns=['지정업종', '에너지 사용량(TJ)'])

# 합계 도출을 위해 object(문자형)데이터를 float(실수)로 변환
# 숫자에 자동으로 삽입된 ',' 없애기
df1['에너지 사용량(TJ)'] = df1['에너지 사용량(TJ)'].str.replace(',', '').astype(float)

# 지정업종별로 그룹을 나누어 합계 도출
a1 = df1.groupby('지정업종').sum()

# 에너지 사용량을 내림차순 기준으로 정렬
b1 = a1.sort_values('에너지 사용량(TJ)', ascending=False)

# 상위 10개 행 추출
c1 = b1.head(10)

# 수평 막대 그래프로 표현하기 위해 y축의 길이 정해주기
y = np.arange(10)

# 단위를 만으로 축약하여 숫자 추출
c1_values = [int(i / 10000) for i in list(c1['에너지 사용량(TJ)'])]

# matplotlib를 활용한 시각화
plt.figure(figsize=(20, 6))
plt.title('업종별 에너지 사용량', loc='center', fontsize=15, fontweight='bold', fontproperties=fp)
plt.title('(단위 : 만)', loc='right', fontsize=12, fontproperties=fp1)
plt.barh(y, c1_values, color=colors_list1, alpha=0.6)
plt.yticks(y, c1.index, fontproperties=fp1)
plt.xlabel('에너지 사용량', fontproperties=fp1, fontsize=15)
plt.ylabel('에너지 사용 상위 10개 업종', fontproperties=fp1, fontsize=15)
for i in range(len(y)):
    plt.text(c1_values[i], y[i], str(c1_values[i]),
             fontsize=11,
             color='black',
             horizontalalignment='left',
             verticalalignment='center'
             )
plt.show()

#########################################################################################
# 지역별 데이터 셋 사용

# 2019지역별.csv데이터셋에서 지역 종류 추출
location = list(datas2.iloc[9:, 1])

# 데이터셋에서 사업장 갯수, 가스 배출량 추출
business = []
gas_exhaust = []

# 숫자가 1000단위 이상일 경우 ,가 섞이기 때문에 제거한 후에 정수형으로 형변환
# 사업장 갯수 추출후 추가
for i in list(datas2.iloc[9:, 3]):
    if len(i) >= 4:
        i = i.split(',')
        i = ''.join(i)
    business.append(int(i))

# 가스 배출량 추출후 추가. 숫자가 큰 관계로 10만 단위로 나누어 추가
for i in list(datas2.iloc[9:, 4]):
    if len(i) >= 4:
        i = i.split(',')
        i = ''.join(i)
    gas_exhaust.append(int(int(i) / 100000))

# 1.바 그래프 형태로 지역별 사업장 갯수 출력
plt.figure(figsize=(20, 6))
plt.barh(location, business, alpha=0.7, color=colors_list2)

# 막대그래프 끝 마디에 갯수 텍스트로 출력 추가
# plt.text는 첫 3개의 인자는 각각 텍스트가 출력될 (x축의 위치, y축의 위치, 출력할 텍스트)이다.
for i in range(len(location)):
    plt.text(business[i], location[i], str(business[i]),
             fontsize=11,
             color='black',
             horizontalalignment='left',
             verticalalignment='center'
             )
plt.title("지역별 사업장 개수", loc='center', fontsize=15, fontweight='bold', fontproperties=fp)
plt.title('(단위 : 개)', loc='right', fontsize=12, fontproperties=fp1)
plt.ylabel('지역 구분', fontproperties=fp1, fontsize=15)
plt.show()

# 2.바 그래프 형태로 지역별 가스 배출량 출력
plt.figure(figsize=(20, 6))
plt.barh(location, gas_exhaust, alpha=0.7, color=colors_list2)

# 그래프 끝 마디에 배출량 텍스트로 출력 추가
# plt.text는 첫 3개의 인자는 각각 텍스트가 출력될 (x축의 위치, y축의 위치, 출력할 텍스트)이다.
for i in range(len(location)):
    plt.text(gas_exhaust[i], location[i], str(gas_exhaust[i]),
             fontsize=11,
             color='black',
             horizontalalignment='left',
             verticalalignment='center'
             )
plt.title("지역별 온실가스 배출량", loc='center', fontsize=15, fontweight='bold', fontproperties=fp)
plt.title('(단위 : 십만(tCO2-eq))', loc='right', fontsize=12, fontproperties=fp1)
plt.ylabel('지역 구분', fontproperties=fp1, fontsize=15)
plt.show()

# 1. 업종별 온실가스 배출량
# matplotlib를 활용한 시각화 - 파이 차트

# 상위 10개 업종의 온실가스 배출 총량 구하기
c_sum = int(c.sum())

# 상위 10개 업종의 온실가스 배출 비율 구하기
top10_gas_ratio = [i / c_sum for i in list(c['온실가스 배출량(tCO2-eq)'])]
labels = c.index

# 2. 업종별 에너지 사용량
# matplotlib를 활용한 시각화 - 파이 차트 그래프

# 상위 10개 업종의 에너지 사용량 총량 구하기
c1_sum = int(c1.sum())

# 상위 10개 업종의 에너지 사용 비율 구하기
top10_eng_ratio = [i / c1_sum for i in list(c1['에너지 사용량(TJ)'])]
labels1 = c1.index

# 파이차트로 지역별 사업체 비율 출력
# 1행 2열의 형태로 2개의 파이 차트 동시에 출력
# auctpct = 설정한 자릿수까지 수치를 표현해준다.
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(18, 6)
axes[0].set_title("상위 10개 업종 가스 배출량 비율", fontproperties=fp, fontsize=15, fontweight='bold')
axes[0].pie(top10_gas_ratio, labels=labels, autopct='%.1f%%',
            startangle=260, counterclock=False, wedgeprops=wedgeprops, colors=colors_list)

axes[1].set_title("상위 10개 업종 에너지 사용비율", fontproperties=fp, fontsize=15, fontweight='bold')
axes[1].pie(top10_eng_ratio, labels=labels1, autopct='%.1f%%',
            startangle=260, counterclock=False, wedgeprops=wedgeprops, colors=colors_list1)
plt.show()

# 파이차트로 지역별 사업체 비율 출력
# 1행 2열의 형태로 2개의 파이 차트 동시에 출력
# auctpct = 설정한 자릿수까지 수치를 표현해준다.

# 전체 사업장 대 각 지역별 사업장 개수 비율, 전체 가스 배출량 대 각 지역별 온실가스 배출량비율  산출
bus_ratio = [i / sum(business) for i in business]
gas_ratio = [i / sum(gas_exhaust) for i in gas_exhaust]

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(18, 6)
axes[0].set_title("지역별 사업체 비율", fontproperties=fp, fontsize=15, fontweight='bold')
axes[0].pie(bus_ratio, labels=location, startangle=260, counterclock=False, autopct='%.1f%%', wedgeprops=wedgeprops,
            colors=colors_list2)

# 파이차트로 지역별 가스 배출량 비율 출력
axes[1].set_title("지역별 온실가스 배출량", fontproperties=fp, fontsize=15, fontweight='bold')
axes[1].pie(gas_ratio, autopct='%.1f%%', labels=location, startangle=200, counterclock=False, wedgeprops=wedgeprops,
            colors=colors_list2)
plt.show()