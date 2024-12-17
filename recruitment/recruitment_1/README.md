# 개요

기업을 통해 확보한 실데이터인 [기업정보], [공고정보], [이력정보] 데이터를 통해서 개인과 기업을 연결시켜주는 추천 인공지능 모델을 구축. 

불필요한 데이터(모집인원 0명)와 비어있는 데이터(자본금,매출현황,사원수가 0이거나 공고 키워드 없음)에 대한 전처리.

[기업정보]와 [공고정보]를 '기업 번호'에 따라 매칭시켜 [기업공고정보]와 [이력정보] 2개의 데이터로 분리.

서로에 대한 적합성 점수에 따라 이진 분류 문제로 접근.


## 0. Installation
```bash
  conda create -n scout
  pip install -r requirements.txt
```

## 1. Data preprocessing
- 불필요한 데이터(모집인원 0명)와 비어있는 데이터(자본금,매출현황,사원수가 0이거나 공고 키워드 없음)에 대한 전처리.
- 이력서 정보와 공고 정보를 모두 텍스트 형테로 변환.
- `file_path`변수에 데이터 경로가 있다고 가정(외부 공유 불가)
```bash
  python csv_to_text_app.py
  python csv_to_text_ent.py
```

## 2. Featur Extraction
- 사전 학습된 Sentence Transformer를 이용하여 텍스트 변환된 문장을 벡터화.
```bash
  python feature_extract.py
  python ent_labeling.py
```

## 3. Train model
- 모델 학습 및 편향 제거 과정
```bash
  python main.py --train_=True --mode <save dir> --gt_path <your gt path> --app_path <your applicant.csv path> --ent_path <your enterprise.csv path>
  python main.py --train_=False --mode <save dir> --gt_path <your gt path> --app_path <your applicant.csv path> --ent_path <your enterprise.csv path>
```

### 결과 예시

| 지표              | 값      |
|-------------------|---------|
| **Test Loss**     | 0.2821  |
| **Test Accuracy** | 0.8852  |
| **Test Precision**| 0.8638  |
| **Test Recall**   | 0.9147  |
| **Test F1 Score** | 0.8885  |

**테스트 결과 저장 완료!**



