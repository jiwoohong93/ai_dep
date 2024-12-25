# 한국어 문화 데이터셋 내 텍스트 편향성 검출

## 프로젝트 개요
본 프로젝트는 한국어 문화 데이터셋을 활용하여 텍스트 편향성을 검출하기 위한 **Named Entity Recognition (NER)** 태스크를 수행합니다.  
사전 학습된 언어 모델 **KoELECTRA v3**를 활용해 높은 성능을 달성하였으며, 모델의 성능은 **Char F1** 92.7, **Entity F1** 91.8로 측정되었습니다.



## 개발 환경
### Docker 환경
모델 개발 및 실행 환경은 Docker를 활용하여 구축되었으며, 아래의 링크에서 이미지를 다운로드할 수 있습니다:
- [DockerHub: jskpop/aidep_culture](https://hub.docker.com/repository/docker/jskpop/aidep_culture/general)

```bash
docker pull jskpop/aidep_culture:latest
```

## 데이터셋
- 데이터셋: **KLUE NER Benchmark**
- 데이터 구성:
  - **Train**: 21,008개 샘플
  - **Validation**: 5,000개 샘플
  - **Test**: 제공된 `test.json` 사용

데이터셋은 아래 링크에서 다운로드할 수 있습니다:  
[KLUE Task 69: Named Entity Recognition](https://klue-benchmark.com/tasks/69/overview/description)




## 실행 방법
### 1. 데이터 준비
다운로드한 KLUE NER 데이터셋의 압축을 풀고, 작업 디렉토리에 저장합니다:



### 2. 학습 실행
아래 명령어를 사용해 학습을 시작합니다.  
사용자의 환경에 따라 `--dataset_path`와 `--model_name` 등을 조정하세요.

```bash
python train.py --dataset_path {dataset_path} \
                --model_name koelectra-v3 \
                --epoch 200 \
                --bs 64
```
#### 주요 파라미터
- `--dataset_path`: 데이터셋 경로 (예: `/workspace/KLUE-NER/klue-ner-v1.1`)
- `--model_name`: 모델 이름 (예: `koelectra-v3`)
- `--epoch`: 학습 반복 횟수 (기본값: 200)
- `--bs`: 배치 크기 (기본값: 64)

### 2. 학습 실행
아래 명령어를 사용해 학습을 시작합니다.  
사용자의 환경에 따라 `--dataset_path`와 `--model_name` 등을 조정하세요.

```bash
python train.py --dataset_path {dataset_path} \
                --model_name koelectra-v3 \
                --epoch 200 \
                --bs 64
```
#### 주요 파라미터
- `--dataset_path`: 데이터셋 경로 (예: `/workspace/KLUE-NER/klue-ner-v1.1`)
- `--model_name`: 모델 이름 (예: `koelectra-v3`)
- `--epoch`: 학습 반복 횟수 (기본값: 200)
- `--bs`: 배치 크기 (기본값: 64)

### 2. 테스트 실행
학습이 완료된 모델을 사용하여 평가를 수행합니다.
아래 명령어에서 `--model_path`에 학습된 모델 파일 경로를 지정하세요.

```bash
python test.py --dataset_path {dataset_path} \
                --model_path {save_model_path} \
                --bs 64
```
#### 주요 파라미터
- `--dataset_path`: 데이터셋 경로 (예: `/workspace/KLUE-NER/klue-ner-v1.1`)
- `--model_path`: 학습된 모델의 저장 경로


## 평가 지표
- **Char F1**: 문자 기반 평가 지표로, 92.7의 성능 달성
- **Entity F1**: 개체(entity) 기반 평가 지표로, 91.8의 성능 달성