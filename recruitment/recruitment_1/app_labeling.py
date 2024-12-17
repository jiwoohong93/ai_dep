import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

# 1. 데이터 불러오기
gt_df = pd.read_csv('../DATA/gt.csv')  # gt.csv에는 '이력서번호', '지원자', '기업', '공고번호' 컬럼이 있다고 가정
enterprise_df = pd.read_parquet('txt_enterprise.parquet')  # '공고번호', 'embedding', 'label' 컬럼 포함
applicant_df = pd.read_parquet('txt_applicant.parquet')  # '이력서번호', 'embedding' 컬럼 포함

# 2. 임베딩 형식 확인 및 변환
def preprocess_embedding(embedding):
    if isinstance(embedding, str):
        try:
            # 임베딩이 문자열로 저장되어 있을 경우 리스트로 변환
            return ast.literal_eval(embedding)
        except:
            return None
    elif isinstance(embedding, list) or isinstance(embedding, np.ndarray):
        return embedding
    else:
        return None

# 임베딩 전처리
applicant_df['embedding'] = applicant_df['embedding'].apply(preprocess_embedding)
enterprise_df['embedding'] = enterprise_df['embedding'].apply(preprocess_embedding)

# 결측치 확인 및 제거
applicant_df = applicant_df.dropna(subset=['embedding'])
enterprise_df = enterprise_df.dropna(subset=['embedding'])

# 모든 임베딩의 길이가 동일한지 확인
def check_embedding_length(df, column_name):
    lengths = df[column_name].apply(len).unique()
    if len(lengths) != 1:
        print(f"Warning: '{column_name}' 컬럼의 임베딩 길이가 일관되지 않습니다: {lengths}")
    else:
        print(f"'{column_name}' 컬럼의 임베딩 길이: {lengths[0]}")

check_embedding_length(applicant_df, 'embedding')
check_embedding_length(enterprise_df, 'embedding')

# 2. 공고번호를 기준으로 gt_df와 enterprise_df 병합
merged_df = pd.merge(gt_df, enterprise_df, on='공고번호', how='left')

# 확인: 병합된 데이터에 결측치가 없는지 확인
if merged_df['embedding'].isnull().any():
    missing_jobs = merged_df[merged_df['embedding'].isnull()]['공고번호'].unique()
    print(f"Warning: 다음 공고번호에 대한 임베딩이 존재하지 않습니다: {missing_jobs}")

# 3. 이력서와 공고 임베딩을 numpy 배열로 변환
# 이력서 임베딩을 딕셔너리로 변환하여 빠르게 접근할 수 있도록 함
applicant_embeddings = dict(zip(applicant_df['이력서번호'], applicant_df['embedding']))

# 4. 새로운 레이블을 저장할 리스트 초기화
new_labels = []

# 5. 각 이력서에 대해 가장 유사한 공고의 레이블 찾기
for resume_id in tqdm.tqdm(applicant_df['이력서번호']):
    # 해당 이력서가 지원한 공고들 가져오기
    supported_jobs = merged_df[merged_df['이력서번호'] == resume_id]
    import ipdb;ipdb.set_trace()
    if supported_jobs.empty:
        # 지원한 공고가 없는 경우
        new_labels.append(None)
        continue
    
    # 이력서 임베딩 가져오기
    resume_embedding = np.array(applicant_embeddings[resume_id]).reshape(1, -1)
    
    # 공고 임베딩 리스트와 레이블 리스트
    job_embeddings = supported_jobs['embedding'].tolist()
    job_labels = supported_jobs['label'].tolist()
    
    job_embeddings = [emb for emb in job_embeddings if isinstance(emb, (list, np.ndarray))]
    # 공고 임베딩이 비어있을 경우 처리
    if not job_embeddings:
        new_labels.append(None)
        continue
    
        # 공고 임베딩을 numpy 배열로 변환
    job_embeddings = np.vstack(job_embeddings)
    
    # 이력서 임베딩을 2차원 배열로 변환
    resume_embedding = resume_embedding.reshape(1, -1)
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(resume_embedding, job_embeddings)[0]
    
    # 가장 높은 유사도를 가진 공고의 인덱스
    max_idx = np.argmax(similarities)
    
    # 해당 공고의 레이블을 새로운 레이블로 할당
    new_label = job_labels[max_idx]
    new_labels.append(new_label)

# 6. 새로운 레이블을 applicant_df에 추가
applicant_df['new_label'] = new_labels

# 7. 결과 확인 (옵션)
print(applicant_df[['이력서번호', 'new_label']].head())

# 8. 업데이트된 applicant_df를 저장
applicant_df.to_parquet('txt_applicant_updated.parquet', index=False)

print("레이블 할당이 완료되었으며, 'txt_applicant_updated.parquet' 파일로 저장되었습니다.")
