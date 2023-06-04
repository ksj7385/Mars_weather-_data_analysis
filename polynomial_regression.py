import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split ,KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 수집
df = pd.read_csv('mars-weather.csv')

# 데이터 확인
print(df.head())

# Unusable data
df = df.drop(['id','terrestrial_date', 'month', 'atmo_opacity', 'wind_speed'], axis=1).select_dtypes(include=[np.number])

# 각 feature의 결측치 개수 확인
print(df.isnull().sum())

# 결측치 처리
df.fillna(df.mean(), inplace=True)

# 데이터 정규화
scaler = MinMaxScaler()
df_scaler = scaler.fit_transform(df)
df_scaler = pd.DataFrame(df_scaler, columns=df.columns)
print(df_scaler)


X = df_scaler.drop('ls',axis=1)
y = df_scaler['ls']

#polynomiral regression
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

mse_scores = [] # MSE 점수 저장하는 리스트
r2_scores = [] # R-squared 점수 저장하는 리스트

#k-fold
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=0)

for train_index, test_index in kf.split(X_poly):
    #훈련 세트와 테스트 세트로 분류
    X_train, X_test = X_poly[train_index], X_poly[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 모델 생성 학습
    reg = LinearRegression()
    reg.fit(X_train,y_train)

    # 모델 예측
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    
    #모델 평가
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train,y_train_pred)
    test_mse = mean_squared_error(y_test,y_test_pred)
    test_r2 = r2_score(y_test,y_test_pred)
    
    #모델 평가 점수 저장 
    mse_scores.append(test_mse)
    r2_scores.append(test_r2)


# 평균 점수 계산
mean_mse = np.mean(mse_scores)
mean_r2 = np.mean(r2_scores)

print(f"Mean MSE: {mean_mse}")
print(f"Mean R-squared: {mean_r2}")
