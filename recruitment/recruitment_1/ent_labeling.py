import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 1. Parquet 파일 불러오기
input_parquet_path = 'txt_enterprise.parquet'  # 입력 Parquet 파일 경로
output_parquet_path = 'txt_enterprise.parquet'  # 출력 Parquet 파일 경로

df = pd.read_parquet(input_parquet_path)

# 2. "embeddings" 컬럼 추출
# embeddings 컬럼이 리스트나 배열 형태로 저장되어 있다고 가정
embeddings = np.stack(df['embedding'].values)

# 3. 데이터 중심(centroid) 계산
centroid = np.mean(embeddings, axis=0)

# 4. PCA를 사용하여 초평면 결정
# 데이터 중심을 원점으로 이동
centered_embeddings = embeddings - centroid

# PCA 수행하여 첫 번째 주성분을 찾음
pca = PCA(n_components=1)
import ipdb;ipdb.set_trace()
pca.fit(centered_embeddings)
w = pca.components_[0]  # 초평면의 정규 벡터

# 5. 데이터 포인트에 레이블 할당
# 각 데이터 포인트를 초평면에 투영
projections = centered_embeddings.dot(w)

# 투영 값의 부호에 따라 레이블 부여 (예: 0 또는 1)
labels = (projections >= 0).astype(int)

# 6. 새로운 "label" 컬럼 추가 및 Parquet 파일로 저장
df['label'] = labels

# 변경된 DataFrame을 새로운 Parquet 파일로 저장
df.to_parquet(output_parquet_path, index=False)

print(f"레이블이 추가된 Parquet 파일이 '{output_parquet_path}' 경로에 저장되었습니다.")
