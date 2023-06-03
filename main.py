import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

# Step 1: 데이터 로드
df = pd.read_csv('mars-weather.csv')

# Step 2: 필요하지 않은 feature 제거 및 숫자형 feature 선택
numeric_df = df.drop(['id', 'month', 'atmo_opacity', 'wind_speed'], axis=1).select_dtypes(include=[np.number])

# Step 3: 각 feature의 분포 확인
print(numeric_df.describe())

# Step 4: 결측치 확인 및 평균값으로 대체
print(numeric_df.isnull().sum())  # 각 feature의 결측치 개수 확인
numeric_df.fillna(numeric_df.mean(), inplace=True)

# Step 5: 이상치 확인 및 제거
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
numeric_df_no_outliers = numeric_df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 6: 데이터 정규화 (min-max normalization)
scaler = MinMaxScaler()
numeric_df_normalized = pd.DataFrame(scaler.fit_transform(numeric_df_no_outliers),
                                     columns=numeric_df_no_outliers.columns)

# Step 7: feature 간의 상관관계 확인
plt.figure(figsize=(12, 6))
sns.heatmap(numeric_df_normalized.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of all features')
plt.show()

# Step 8: 계절을 대표하는 새로운 feature 생성
numeric_df_normalized['season_feature'] = numeric_df_normalized['min_temp'] + numeric_df_normalized['max_temp'] + \
                                          numeric_df_normalized['pressure'] + numeric_df_normalized['sol'] + \
                                          numeric_df_normalized['ls']




# Step 9: K-fold cross validation을 위한 준비
kf = KFold(n_splits=5, shuffle=True, random_state=0)

kmeans_silhouette_scores = []
kmeans_cohesion_scores = []
kmeans_separation_scores = []

agg_silhouette_scores = []
agg_cohesion_scores = []
agg_separation_scores = []

pca_kmeans_silhouette_scores = []
pca_kmeans_cohesion_scores = []
pca_kmeans_separation_scores = []

pca_agg_silhouette_scores = []
pca_agg_cohesion_scores = []
pca_agg_separation_scores = []

# 모델 정보를 저장할 리스트
models_info = []

# Step 10: K-fold cross validation을 위한 준비
for train_index, test_index in kf.split(numeric_df_normalized):
    X_train, X_test = numeric_df_normalized.iloc[train_index], numeric_df_normalized.iloc[test_index]

    # PCA 차원 축소
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Step 10: 클러스터링
    # 무작위 매개 변수를 사용하여 K-Means를 10회 반복
    for _ in range(10):
        kmeans_params = {
            'n_clusters': 4,
            'n_init': np.random.randint(5, 15),
            'random_state': np.random.randint(0, 100)
        }
        kmeans = KMeans(**kmeans_params)

        # K-Means 클러스터링 수행
        kmeans.fit(X_train)
        kmeans_labels = kmeans.predict(X_test)


        silhouette_avg = silhouette_score(X_test, kmeans_labels)
        cohesion = kmeans.inertia_
        separation = cdist(X_test, kmeans.cluster_centers_, 'euclidean').min(axis=1).sum()

        # 모델 정보 저장
        models_info.append({
            'model': 'KMeans',
            'params': kmeans_params,
            'silhouette_score': silhouette_avg,
            'cohesion': cohesion,
            'separation': separation,
            'labels': kmeans_labels,
            'X_test': X_test
        })

        # PCA를 사용한 K-Means 클러스터링 수행
        kmeans.fit(X_train_pca)
        kmeans_pca_labels = kmeans.predict(X_test_pca)


        silhouette_avg = silhouette_score(X_test_pca, kmeans_pca_labels)
        cohesion = kmeans.inertia_
        separation = cdist(X_test_pca, kmeans.cluster_centers_, 'euclidean').min(axis=1).sum()

        # 모델 정보 저장
        models_info.append({
            'model': 'PCA KMeans',
            'params': kmeans_params,
            'silhouette_score': silhouette_avg,
            'cohesion': cohesion,
            'separation': separation,
            'labels': kmeans_pca_labels,
            'X_test': X_test_pca
        })

        metric_options = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
        linkage_options = ['ward', 'complete', 'average', 'single']

        # 무작위 매개 변수를 사용하여 Agglomerative 클러스터링을 10회 반복
        for _ in range(10):
            agg_params = {
                'n_clusters': 4,
                'metric': np.random.choice(metric_options),
                'linkage': np.random.choice(linkage_options)
            }

            # 'ward' linkage는 'euclidean' affinity만 가능하므로 호환되는지 확인
            if agg_params['linkage'] == 'ward' and agg_params['metric'] != 'euclidean':
                continue

            agg_cluster = AgglomerativeClustering(**agg_params)
            agg_labels = agg_cluster.fit_predict(X_test)



            silhouette_avg = silhouette_score(X_test, agg_labels)

            # AgglomerativeClustering에서는 cohesion과 separation을 수동으로 계산
            cohesion = pdist(X_test[agg_labels == 0]).sum()
            separation = pdist(X_test, metric='euclidean').min()

            # 모델 정보 저장
            models_info.append({
                'model': 'Agglomerative',
                'params': agg_params,
                'silhouette_score': silhouette_avg,
                'cohesion': cohesion,
                'separation': separation,
                'labels': agg_labels,
                'X_test': X_test
            })

            # PCA를 사용한 Agglomerative 클러스터링 수행
            agg_pca_labels = agg_cluster.fit_predict(X_test_pca)



            silhouette_avg = silhouette_score(X_test_pca, agg_pca_labels)

            # AgglomerativeClustering에서는 cohesion과 separation을 수동으로 계산
            cohesion = pdist(X_test_pca[agg_pca_labels == 0]).sum()
            separation = pdist(X_test_pca, metric='euclidean').min()

            # 모델 정보 저장
            models_info.append({
                'model': 'PCA Agglomerative',
                'params': agg_params,
                'silhouette_score': silhouette_avg,
                'cohesion': cohesion,
                'separation': separation,
                'labels': agg_pca_labels,
                'X_test': X_test_pca
            })

# 그리드 크기 조정
grid_size = 7
plt.figure(figsize=(15, 15))

# 각 모델을 플로팅
for i, model_info in enumerate(models_info):
    # 만약 모델의 수가 그리드 크기보다 크면 일단 그리드 크기까지만 출력
    if i >= grid_size * grid_size:
        break

    # 시각화를 위한 데이터 전처리
    if isinstance(model_info['X_test'], pd.DataFrame) and model_info['X_test'].shape[1] > 2:
        pca = PCA(n_components=2)
        X_test_np = pca.fit_transform(model_info['X_test'].values)
    else:
        X_test_np = model_info['X_test']

    plt.subplot(grid_size, grid_size, i + 1)
    plt.scatter(X_test_np[:, 0], X_test_np[:, 1], c=model_info['labels'], s=5, cmap='viridis')

# 출력
plt.tight_layout()
plt.show()

# 실루엣 점수를 기준으로 상위 5개 모델을 선택
models_info.sort(key=lambda x: x['silhouette_score'], reverse=True)

top_models = models_info[:5]

tsne = TSNE(n_components=2, random_state=42)

# 산점도를 플로팅
for i, model_info in enumerate(top_models):
    # 서브플롯
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 산점도
    axes[0].scatter(model_info['X_test'][:, 0], model_info['X_test'][:, 1], c=model_info['labels'], s=50,
                    cmap='viridis')
    axes[0].set_title(f"Rank {i + 1}: {model_info['model']}\n "
                      f"with params: {model_info['params']}\n"
                      f"Silhouette Score: {model_info['silhouette_score']:.2f}")

    # t-SNE를 데이터에 적용
    X_tsne = tsne.fit_transform(model_info['X_test'])

    # t-SNE 산점도
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=model_info['labels'], s=50, cmap='viridis')
    axes[1].set_title(f"t-SNE Result for Model {i + 1}")

plt.tight_layout()
plt.show()

# 가장 좋은 모델 선정
best_model_info = top_models[0]

# 모델 생성
if best_model_info['model'] == 'KMeans':
    best_model = KMeans(**best_model_info['params'])
elif best_model_info['model'] == 'PCA KMeans':
    best_model = KMeans(**best_model_info['params'])
    numeric_df_normalized = PCA(n_components=2).fit_transform(numeric_df_normalized)
elif best_model_info['model'] == 'Agglomerative':
    best_model = AgglomerativeClustering(**best_model_info['params'])
elif best_model_info['model'] == 'PCA Agglomerative':
    best_model = AgglomerativeClustering(**best_model_info['params'])
    numeric_df_normalized = PCA(n_components=2).fit_transform(numeric_df_normalized)
else:
    raise ValueError(f"Unknown model type: {best_model_info['model']}")

# 가장 좋은 모델로 전체 데이터 학습 및 클러스터링
labels = best_model.fit_predict(numeric_df_normalized)

# 결과 산점도 그리기
plt.figure(figsize=(8, 8))
plt.scatter(numeric_df_normalized[:, 0], numeric_df_normalized[:, 1], c=labels, s=15, cmap='viridis')
plt.title(f"Best Model: {best_model_info['model']}\n"
          f"Params: {best_model_info['params']}\n"
          f"Silhouette Score: {best_model_info['silhouette_score']:.2f}")
plt.show()



# 상위 5개 모델 정보 콘솔 출력
for i, model_info in enumerate(top_models):
    print("\n---\n")
    print(f"Model Rank: {i+1}")
    print(f"Model Type: {model_info['model']}")
    print(f"Parameters: {model_info['params']}")
    print(f"Silhouette Score: {model_info['silhouette_score']:.2f}")
    print(f"Cohesion: {model_info['cohesion']}")
    print(f"Separation: {model_info['separation']}")
