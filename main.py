import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

# 데이터 로드
df = pd.read_csv('mars-weather.csv')

# 'id', 'month', 'atmo_opacity', 'wind_speed' feature 제외
numeric_df = df.drop(['id', 'month', 'atmo_opacity', 'wind_speed'], axis=1).select_dtypes(include=[np.number])

# 1. 각 feature의 분포 확인
print(numeric_df.describe())

# 2. 결측치 확인 및 처리
print(numeric_df.isnull().sum())  # 각 feature의 결측치 개수 확인

# 결측치를 평균값으로 대체 (다른 방법으로 중앙값이나 최빈값을 사용할 수도 있습니다)
numeric_df.fillna(numeric_df.mean(), inplace=True)

# 결측치를 중앙값으로 대체 (주석 처리)
# numeric_df.fillna(numeric_df.median(), inplace=True)

# 결측치를 최빈값으로 대체 (주석 처리)
# numeric_df.fillna(numeric_df.mode().iloc[0], inplace=True)

# 3. 이상치 확인 (이상치 제거 전)
fig, axs = plt.subplots(len(numeric_df.columns), 3, figsize=(10, 2 * len(numeric_df.columns)))
for i, column in enumerate(numeric_df.columns):
    numeric_df.boxplot(column, ax=axs[i, 0])
    axs[i, 0].set_title(f'Initial {column}')

# 이상치 제거
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
numeric_df_no_outliers = numeric_df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

# 이상치 확인 (이상치 제거 후, 정규화 전)
for i, column in enumerate(numeric_df_no_outliers.columns):
    numeric_df_no_outliers.boxplot(column, ax=axs[i, 1])
    axs[i, 1].set_title(f'After Outlier Removal {column}')

# 데이터 정규화 (min-max normalization)
scaler = MinMaxScaler()
numeric_df_normalized = pd.DataFrame(scaler.fit_transform(numeric_df_no_outliers),
                                     columns=numeric_df_no_outliers.columns)

# 데이터 정규화 (mean normalization, 주석 처리)
# numeric_df_normalized = (numeric_df_no_outliers - numeric_df_no_outliers.mean()) / (numeric_df_no_outliers.max() - numeric_df_no_outliers.min())

# 데이터 정규화 (standardization, 주석 처리)
# scaler = StandardScaler()
# numeric_df_normalized = pd.DataFrame(scaler.fit_transform(numeric_df_no_outliers), columns=numeric_df_no_outliers.columns)

# 데이터 정규화 (euclidean distance, 주석 처리)
# numeric_df_normalized = numeric_df_no_outliers / np.sqrt((numeric_df_no_outliers**2).sum(axis=0))

# 이상치 확인 (이상치 제거 후, 정규화 후)
for i, column in enumerate(numeric_df_normalized.columns):
    numeric_df_normalized.boxplot(column, ax=axs[i, 2])
    axs[i, 2].set_title(f'After Normalization {column}')

plt.tight_layout()
plt.show()

# 4. feature 간의 상관관계 확인
plt.figure(figsize=(12, 6))
sns.heatmap(numeric_df_normalized.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of all features')
plt.show()

# 계절을 대표하는 새로운 feature 생성
numeric_df_normalized['season_feature'] = numeric_df_normalized['min_temp'] + numeric_df_normalized['max_temp'] + \
                                          numeric_df_normalized['pressure'] + numeric_df_normalized['sol'] + \
                                          numeric_df_normalized['ls']

# K-means 클러스터링
kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)  # 4개의 계절을 가정

# K-fold cross validation을 위한 준비
kf = KFold(n_splits=5, shuffle=True, random_state=0)

silhouette_scores = []  # 실루엣 점수를 저장할 리스트
cohesion_scores = []  # 클러스터 응집성을 저장할 리스트
separation_scores = []  # 클러스터 분리도를 저장할 리스트

for train_index, test_index in kf.split(numeric_df_normalized):
    train_data = numeric_df_normalized.loc[train_index]
    test_data = numeric_df_normalized.loc[test_index]

    # K-means 클러스터링
    kmeans_clusters = kmeans.fit_predict(train_data[['season_feature']])

    # 클러스터 결과를 데이터프레임에 'season_kmeans' feature로 추가
    train_data['season_kmeans'] = kmeans_clusters

    # 실루엣 점수 계산
    silhouette = silhouette_score(train_data[['season_feature']], kmeans_clusters)
    silhouette_scores.append(silhouette)

    # 클러스터 응집성 계산
    cohesion = np.mean(pairwise_distances(train_data[['season_feature']], metric='euclidean'))
    cohesion_scores.append(cohesion)

    # 클러스터 분리도 계산
    separation = 0
    for i in range(4):
        cluster_data = train_data[train_data['season_kmeans'] == i][['season_feature']]
        other_clusters_data = train_data[train_data['season_kmeans'] != i][['season_feature']]
        separation += np.mean(pairwise_distances(cluster_data, other_clusters_data, metric='euclidean'))
    separation /= 4
    separation_scores.append(separation)

    # Hierarchical Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=4)
    agg_clusters = agg_clustering.fit_predict(train_data[['season_feature']])

    # 클러스터 결과를 데이터프레임에 'season_agg' feature로 추가
    train_data['season_agg'] = agg_clusters

    # PCA를 사용하여 새로운 feature 생성
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(train_data[['min_temp', 'max_temp', 'pressure', 'sol', 'ls']])

    # K-means 클러스터링 결과 시각화
    plt.figure(figsize=(10, 6))
    for i in range(4):
        cluster = pca_features[train_data['season_kmeans'] == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Season {i}')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.title('Season Clusters by K-means Clustering')
    plt.legend()
    plt.show()

    # Hierarchical Agglomerative Clustering 결과 시각화
    plt.figure(figsize=(10, 6))
    for i in range(4):
        cluster = pca_features[train_data['season_agg'] == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Season {i}')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.title('Season Clusters by Hierarchical Agglomerative Clustering')
    plt.legend()
    plt.show()

# 평균 실루엣 점수 출력
mean_silhouette = np.mean(silhouette_scores)
print('Mean Silhouette Score:', mean_silhouette)

# 평균 클러스터 응집성 출력
mean_cohesion = np.mean(cohesion_scores)
print('Mean Cohesion Score:', mean_cohesion)

# 평균 클러스터 분리도 출력
mean_separation = np.mean(separation_scores)
print('Mean Separation Score:', mean_separation)
