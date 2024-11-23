import streamlit as st
import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 페이지 타이틀 설정
st.set_page_config(page_title="머신 러닝 앱",page_icon='🤖', layout="wide")

st.title('💻 머신 러닝 앱')

st.info('머신러닝 모델을 구축하는 앱 입니다!')

# @st.cache_data
def load_data():
    return pd.read_csv("/content/lecture12.csv")

df = load_data()

# Expander (확장자) 사용
with st.expander('Data'):
  st.write('**Raw data(원본 데이터)**')
  df.head()

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  st.dataframe(X_raw)

  st.write('**y**')
  y_raw = df['species']
  y_raw # st.dataframe 불필요 (st.dataframe 없이 표현 가능)

with st.expander('Data visualization'):
  options = st.multiselect(
      "시각화를 희망하는 변수를 선택하세요",
      ["bill_length_mm", "bill_depth_mm","flipper_length_mm",	"body_mass_g"],
      ["bill_length_mm", "bill_depth_mm"] )
  if len(options) >= 2:
        st.scatter_chart(
            data=df,
            x=options[0],
            y=options[1],
            color='species'
        )
  else:
        st.warning("적어도 두 개의 변수를 선택하세요.")




# 입력 변수 (사이드바 부분)
with st.sidebar:
  st.header('입력변수 창')
  island_mapper = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  island = island_mapper[island]
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender_mapper = {'male': 1, 'female': 0}
  gender = st.selectbox('Gender', ('male', 'female'))
  gender = gender_mapper[gender]

  # 입려된 값으로 데이터 프레임 생성
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])


  X_raw['island'] =  X_raw['island'].replace(island_mapper)
  X_raw['sex'] =  X_raw['sex'].replace(gender_mapper)

  input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('입력 데이터'):
  st.write('**예측하고자 하는 값**')
  input_df
  st.write('**전체 데이터**')
  input_penguins


# 데이터 변환
# X 변수 처리
# encode = ['island', 'sex']
# df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = input_penguins[1:]
input_row = input_penguins[:1]

# y 변수 처리
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}

def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y


# 모델 학습 및 예측
## 모델의 학습
clf = RandomForestClassifier()
clf.fit(X, y)

## 학습 모델을 활용한 예측값 도출
prediction = clf.predict(input_row)
# st.write(prediction)
## 단순 분류를 넘어, 확률을 도출
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})

# 결과값 시각화
# 일종의 스타일 가이드
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
