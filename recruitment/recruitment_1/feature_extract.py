import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# GPU 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 가능한 디바이스: {device}")

# 모델 로드 및 디바이스 설정
model_name = 'jhgan/ko-sroberta-multitask'
model = SentenceTransformer(model_name)
model.to(device)  # 모델을 GPU로 이동

# CSV 파일 경로 (여러 파일을 처리하기 위해 리스트로 관리)
csv_files = ['../DATA/txt_applicant.csv', '../DATA/txt_enterprise.csv']  # 실제 파일명으로 변경

# Parquet 저장 경로 설정
parquet_files = ['txt_applicant.parquet', 'txt_enterprise.parquet']

# 배치 사이즈 설정 (GPU 메모리에 따라 조정)
batch_size = 64

for csv_file, parquet_file in zip(csv_files, parquet_files):
    print(f"\n{csv_file} 처리 중...")

    # 데이터 로드
    if csv_file.split("_")[-1].startswith("a"):
        continue
        #data = pd.read_csv(csv_file, usecols=['이력서번호', '지원자번호', '설명'])
    else:
        data = pd.read_csv(csv_file, usecols=['기업번호', '공고번호', '설명'])

    # "설명" 컬럼 리스트로 추출
    descriptions = data['설명'].tolist()

    # 임베딩 생성
    embeddings = []

    print("임베딩 생성 중...")
    for i in tqdm(range(0, len(descriptions), batch_size)):
        batch = descriptions[i:i + batch_size]
        batch_embeddings = model.encode(batch, batch_size=batch_size, device=device, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    # 임베딩 추가
    data['embedding'] = embeddings

    # Parquet 파일로 저장
    print(f"Parquet 파일로 저장 중: {parquet_file}")
    data.to_parquet(parquet_file, index=False)

    print(f"{csv_file} -> {parquet_file} 저장 완료!")
