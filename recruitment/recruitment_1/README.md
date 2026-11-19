# 채용 1: 이력서-공고 매칭 및 개인공정성 검증

SCOUT 출처의 이력사항, 기업·공고정보, 지원내역을 연결하여 이력서-공고 매칭을 예측하고 성별 반전 전후의 추천 기업 일치율을 측정합니다. 현재 버전은 KURE-v1 임베딩, MLP 매칭 모델, AlphaEdit에서 착안한 출력층 보정을 사용합니다.

기존 `main.py` 기반 구현은 유지했습니다. 이전 설명과 환경 목록은 [README_legacy.md](README_legacy.md), [requirements_legacy.txt](requirements_legacy.txt)를 참고하세요. 아래 실행 방법과 결과는 `run_experiment.py` 기반 현재 버전에 해당합니다.

## 1. 검증 결과

| 지표 | 보정 전 | 보정 후 | 2단계 목표 |
|---|---:|---:|---:|
| 매칭 분류 정확도 (%) | 92.5384 | 92.5422 | 80 이상 |
| 개인공정성: 최상위 추천 기업 일치율 (%) | 99.4654 | 99.5525 | 95 이상 |
| 성별 반전 후 추천 기업 변경 | 43 / 8,044 | 36 / 8,044 | - |

집계 결과는 [results/metrics.json](results/metrics.json)에 있습니다. 자체 실데이터 검증 결과이며, 공인시험 성적서를 의미하지 않습니다.

## 2. 데이터와 입력·출력

원본은 비공개 XLSX 3개, 시트 4개입니다.

| 파일 | 시트 | 사용 정보 |
|---|---|---|
| 스카우트_기업회원DB_20221222.xlsx | 기업정보 | 기업 식별자, 기업명, 업종, 기업형태, 자본금, 매출, 사원수 |
| 동일 파일 | 공고정보 | 기업·공고 식별자, 공고명, 직종, 키워드, 경력·학력·근무조건 |
| 스카우트_이력사항DB_20221222.xlsx | 통합정보 | 이력서·지원자 식별자, 희망직종, 경력, 담당업무, 학력·학교·전공, 어학·자격사항 등 |
| 스카우트_지원내역DB_20221222.xlsx | 지원내역 | 이력서·지원자·기업·공고 식별자를 통한 실제 지원 관계 |

- 이력사항은 이력서·지원자 단위, 공고는 기업·공고 단위로 반복 값을 통합합니다.
- 실제 지원 쌍을 positive로 사용합니다. 같은 이력서의 지원 이력을 제외한 공고를 무작위로 추출하여 동수의 negative를 구성합니다.
- 학습 정답은 지원 여부에 기반한 쌍 라벨입니다.
- 이력서·공고 텍스트를 각각 정규화된 1,024차원 임베딩으로 변환합니다. 입력 길이는 512 토큰으로 제한합니다.
- MLP는 두 임베딩의 투영값·차이·곱·코사인 유사도와 이진 성별 평가 라벨을 사용합니다.
- 출력은 쌍별 매칭 확률과 0.5 기준의 이진 예측입니다. 동일 이력서의 후보 공고 점수를 비교하여 최상위 기업을 정합니다.
- 전체 입력 컬럼과 텍스트 구성은 `run_experiment.py`의 `prepare`, `build_applicants`, `build_jobs`에 정의되어 있습니다.

| 데이터 구분 | 수 |
|---|---:|
| 원본 지원내역 | 266,003 |
| 정제 후 positive / 무작위 negative | 각각 262,274 |
| 전체 실험 쌍 | 524,548 |
| Train / Validation / Test | 419,638 / 52,455 / 52,455 |
| 실험 쌍에 포함된 지원자 / 이력서 | 27,401 / 32,588 |
| 실험 쌍에 포함된 공고 / 기업 | 217,030 / 14,683 |

라벨 비율을 유지하는 쌍 단위 80/10/10 무작위 분할이며 난수 시드는 42입니다. 전처리 후 전체 이력서·공고 테이블은 각각 352,893개, 379,506개입니다. 전체 테이블 크기와 실험 쌍에 포함된 고유 개체 수는 구분합니다.

## 3. 지표와 보정 방법

**매칭 분류 정확도**는 테스트 52,455쌍에서 정답과 이진 예측이 일치한 비율입니다.

**개인공정성 지표**는 동일 이력서와 테스트 후보 집합을 고정하고 성별 평가 라벨을 반전했을 때, 최고 점수 공고의 기업이 유지되는 비율입니다.

- 지원자별로 이진 성별 평가 라벨을 무작위 부여하고 같은 지원자에 대해 유지합니다.
- 평가 시 라벨을 `g -> 1-g`로 변경하며 나머지 입력은 고정합니다.
- 주 지표의 분모는 테스트 후보 집합에 서로 다른 기업이 두 곳 이상 있는 이력서 8,044개입니다.
- 테스트에 연결된 후보 공고를 비교하며 전체 공고 카탈로그를 검색하는 평가는 아닙니다.
- 기업 점수는 후보 공고 중 최고 점수 공고가 속한 기업으로 정합니다. 동점에서는 테이블의 첫 번째 최고 점수 공고를 선택합니다.

출력층 보정은 성별 반전 입력의 예측과 기본 모델의 원래 예측 사이 차이를 줄이고, 원래 출력 보존 손실 및 지도학습 손실을 함께 사용합니다. 기본 모델 표현의 상위 128개 방향으로부터 구한 보완 공간에 출력층 기울기를 투영합니다. 표현부와 출력층 bias는 고정하며 출력층 weight만 AdamW로 조정합니다. 세부 설정은 [configs/experiment.json](configs/experiment.json)에 고정했습니다.

## 4. 환경 설치

검증 환경은 Python 3.10.20, PyTorch 2.9.1 + CUDA 12.8입니다. 임베딩은 GPU 4개, MLP 학습·보정·평가는 GPU 1개를 사용했습니다. 정확한 실행 환경과 재현 확인 결과는 [results/verification_environment.json](results/verification_environment.json)에 기록했습니다. 2026년 11월 19일 동일 환경에서 기존 임베딩으로 재학습한 결과, 두 모델의 파라미터와 집계 지표가 기준 모델과 일치함을 확인했습니다.

```bash
conda create -n recruitment1 python=3.10.20 -y
conda activate recruitment1
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

모델 다운로드에는 인터넷 연결이 필요합니다. 임베딩 모델 revision도 설정 파일에 고정했습니다. CPU 평가도 가능하지만 연산 정밀도·장치·배치 크기가 달라지면 경계 예측이나 동점 처리에 따른 수치 차이가 생길 수 있습니다. 기준 결과 대조에는 검증 환경과 동일한 CUDA 평가 설정을 사용하세요.

## 5. 고정 모델 평가: 재학습 없음

원본·파생 데이터, 임베딩, 학습된 가중치는 공개 저장소에 포함하지 않습니다. 승인된 평가자는 다음 파일을 별도 경로에 준비해야 합니다.

```text
PRIVATE_ARTIFACT_DIR/
  applicants.parquet
  jobs.parquet
  samples.parquet
  applicants_embeddings.npy
  jobs_embeddings.npy
  base_model.pt
  alphaedit_fair_model.pt
```

보관된 기준 파일의 해시는 `PRIVATE_ARTIFACT_SHA256SUMS`에 있습니다. 파일을 제공받은 후 해당 디렉토리에서 `sha256sum -c /path/to/recruitment_1/PRIVATE_ARTIFACT_SHA256SUMS`로 무결성을 확인할 수 있습니다.

```bash
cd recruitment/recruitment_1
bash cmd.sh evaluate /path/to/PRIVATE_ARTIFACT_DIR
```

평가는 두 체크포인트를 로드하여 원래 성별·반전 성별에 대해 각각 추론합니다. 샘플별 개인정보를 출력하지 않고 집계 결과만 기록합니다.

- 기준 샘플 수와 `results/metrics.json`의 수치를 허용 오차 `1e-6`로 대조합니다.
- 정확도 80%, 개인공정성 95% 목표 달성 여부를 확인합니다.
- 결과를 `PRIVATE_ARTIFACT_DIR/evaluation_metrics.json`에 저장합니다.
- 수치 불일치 또는 목표 미달 시 0이 아닌 종료 코드를 반환합니다.

출력 마지막 두 줄은 다음과 같습니다.

```text
Reference metrics: PASS
Phase-2 targets: PASS
```

평가만 수행하면 원래 체크포인트와 전처리 파일은 변경하지 않습니다. 낮은 VRAM 환경에서는 `python evaluate.py --artifact-dir ... --batch-size 4096 --check-targets`처럼 배치를 조절할 수 있습니다.

## 6. 전처리부터 재학습

승인된 원본 XLSX 3개를 별도 디렉토리에 준비합니다. 생성 데이터와 모델은 공개 소스 디렉토리 밖에 저장하는 것을 권장합니다.

```bash
bash cmd.sh all /path/to/NEW_ARTIFACT_DIR /path/to/PRIVATE_XLSX_DIR
```

전처리·임베딩이 이미 있다면 다음 명령으로 기본 모델 학습과 공정성 보정만 다시 수행합니다. 출력 경로의 체크포인트를 새로 작성하므로 보관 모델과 다른 작업 디렉토리를 사용하세요.

```bash
bash cmd.sh train /path/to/NEW_ARTIFACT_DIR
python evaluate.py --artifact-dir /path/to/NEW_ARTIFACT_DIR --check-targets
```

개별 단계 및 설정 변경도 지원합니다. 명시한 CLI 옵션이 JSON 설정보다 우선합니다.

```bash
python run_experiment.py --config configs/experiment.json --stage prepare --data-dir /path/to/PRIVATE_XLSX_DIR --out-dir /path/to/NEW_ARTIFACT_DIR
python run_experiment.py --config configs/experiment.json --stage embed --out-dir /path/to/NEW_ARTIFACT_DIR
python run_experiment.py --config configs/experiment.json --stage train --out-dir /path/to/NEW_ARTIFACT_DIR
```

학습 과정은 train/validation을 사용하며 테스트는 결과 집계에 사용합니다. 재학습 결과는 장치와 라이브러리의 수치 연산에 따라 차이가 날 수 있습니다. 고정 체크포인트 결과의 재현과 재학습 결과의 목표 성능 달성은 별도로 확인합니다.

## 7. 공개 범위와 테스트

현재 버전에서 공개하는 자료는 코드, 고정 설정, 집계 지표, 학습 이력, 무결성 해시입니다. 원본 XLSX, 이력서 원문, 식별자가 담긴 파생 테이블, 임베딩, 학습된 가중치와 실제 입력 예시는 새로 업로드하지 않습니다. 기존 버전의 추적 파일은 별도로 유지합니다.

공개 데이터 없이 합성 데이터로 실행하는 테스트:

```bash
python -m unittest discover -s tests -v
bash -n cmd.sh
```

이 테스트는 전처리, 분할·성별 라벨 일관성, 지표의 인덱스 정렬과 분모, 설정 우선순위, 비공개 체크포인트 평가 경로를 확인합니다.



