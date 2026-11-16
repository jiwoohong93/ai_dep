# ALINJob V2.0

채용 지원자 이력 정보(SCOUT 이력사항 DB)로 **최종학력(고졸·전문대·대졸·석사·박사 5분류)** 을 분류하고,
**소수 클래스 선택 가중(minority floor)** 으로 공정성 정책을 반영하는 AI 모델입니다.
ALINJob V1.0(RandomForest, `train_fair.py`·`train_unfair.py`·`test.py`)의 후속 버전입니다.

| 파일 | 내용 |
|---|---|
| `preprocess.py` | 원본 xlsx → `data_stage1.csv` (1단계 전처리, 59개 열, 9,991건) |
| `train_edu_gbm.py` | LightGBM 학습 (정책 반영 `minority` / 정책 미반영 `none`) → `models/trained_edu_<fairness>.pkl` |
| `test_edu_gbm.py` | 평가: 5-fold OOF 정확도·공정성(DEO)·정책 반영 성능 평균 하락률 |
| `cmd.sh` | 데이터 생성 → Hash 확인 → 학습 → 학습 모델 Hash 확인 → 평가 전체 재현 스크립트 |
| `SHA256SUMS` | 시험 데이터·코드 SHA-256 (공인시험 규격서와 동일) |
| `MODEL_SHA256SUMS` | 학습 모델 파일 SHA-256 (결정적 학습, 공인시험 규격서와 동일) |
| `requirements.txt`, `environment.txt` | 고정 버전 / 실행 환경 전체 패키지 목록 |
| `results/` | 재현 실행 로그 (학습 `train_*.log`, 평가 `test_skip_point.log`·`test_full.log`) |
| `train_fair.py`, `train_unfair.py`, `test.py` | ALINJob V1.0 (참고용, 유지) |

---

## 방법

- **모델**: LightGBM (Gradient Boosting Decision Tree, num_leaves=63). 평점·입학/졸업년도·재학기간·전공·계열·희망/최종연봉·경력 기간·희망 업종/직종 등
  이력 정보를 입력으로 사용하며, 정답과 1:1 대응되는 열(학력구분)과 행 식별용 열(이력서번호 등)은 제외합니다.
  학교명은 **입력 특징으로 쓰지 않고** 공정성 평가용 보호 속성(‘여자’ 포함 학교 → 여성)으로만 사용합니다.
- **학습 설정**: `LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=80,
  subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=1.0)`, 학습용 데이터의 10 %를 검증용으로 분할해
  조기 종료(60 회)를 적용합니다.
- **정책 반영 (minority)**: 학습 데이터에서 표본 수가 floor(=250) 미만인 클래스에만 `floor / count` 가중치를 부여하고,
  다수 클래스는 그대로 둡니다. 정책 미반영(none)은 가중 없이 학습합니다.
- **평가**: StratifiedKFold(k=5, shuffle, seed 42)로 모든 데이터에 대해 학습에 사용하지 않은 fold의 예측(OOF)을 만들고,
  전처리 상태(희귀 범주, 범주 사전, 평점 스케일)는 fold별 학습 데이터에서만 적합합니다(정보 누수 없음).
  두 모델은 동일한 데이터 분할과 동일한 입력 특징을 사용합니다.

| 지표 | 정의 | 기준 |
|---|---|---|
| 분류 정확도 (%) | N_correct / N_total × 100 | > 85 |
| 공정성 DEO | \|TPR_남 − TPR_여\| (고졸 클래스, V1.0 `test.py`와 동일 정의) | < 0.07 |
| 정책 반영 성능 평균 하락률 (%) | \|ACC_fair − ACC_unfair\| × 100 | < 5 |

## 결과 (5-fold OOF, 9,991건)

| 모델 | 정확도 | DEO | 정책 반영 성능 평균 하락률 |
|---|---|---|---|
| 정책 미반영 (none) | 86.04 % (8,596건) | – | – |
| **정책 반영 (minority, floor=250)** | **85.98 %** (8,590건) | **0.0136** | **0.060 %** |

학습은 모든 난수 시드(42)를 고정하고 LightGBM을 단일 스레드 결정적 모드(`n_jobs=1, deterministic=True, force_row_wise=True`)로 실행하며,
특징 상태도 해시 순서와 무관하게 저장하므로 학습 모델 파일이 byte 단위로 동일하게 재현됩니다.
masternode(Ubuntu 24.04, Xeon Gold 6426Y)와 alin11(Ubuntu 22.04, Xeon Silver 4214)에서
데이터·학습 모델 Hash와 평가 결과가 모두 동일함을 확인했습니다.

---

## 실행

```bash
conda create -n ai_dep python=3.10.20 -y && conda activate ai_dep
pip install -r requirements.txt
bash cmd.sh            # 데이터 생성 → 학습 → 평가 (data_stage1.csv 또는 scout_data/ 필요)
```

공인시험(KGcT) 시험 절차 — `data_stage1.csv`만 있으면 실행 가능 (CPU, 학습 약 10 초 + 평가 약 20 초):

```bash
sha256sum -c SHA256SUMS                     # 데이터·코드 무결성 확인
rm -f models/trained_edu_none.pkl models/trained_edu_minority.pkl
python train_edu_gbm.py --data data_stage1.csv --fairness none
python train_edu_gbm.py --data data_stage1.csv
sha256sum -c MODEL_SHA256SUMS               # 학습 모델 Hash 확인 (byte 단위 동일)
python test_edu_gbm.py --data data_stage1.csv --skip-point
```

예상 출력(시각 제외):

```
data: 9991 rows | 고졸 total=154 | protected(여자) 고졸=7
--- ROBUST estimate (full out-of-fold, protected 고졸 support=7) ---
  fair acc=0.8598  unfair acc=0.8604  TPR-=0.129 TPR+=0.143
  분류 정확도 (%)         :  85.977   (> 85)   [PASS]
  공정성 (DEO)            :  0.0136   (< 0.07) [PASS]
  정책 반영 평균 하락률 (%) :   0.060   (< 5)    [PASS]
OVERALL: ALL THRESHOLDS PASS
```

`--skip-point`를 빼면 저장 모델(`models/*.pkl`)의 고정 10 % 분할(seed 42) 평가가 추가로 출력됩니다.

## 데이터

SCOUT 이력사항 DB(2022.12.22. 기준)는 개인정보를 포함한 **비공개 데이터**이므로 원본(`scout_data/*.xlsx`),
파생 데이터(`data_stage1.csv`), 학습 모델(`models/*.pkl`)은 저장소에 포함하지 않습니다(학습 코드로 동일하게 재생성 가능).
데이터를 제공받은 경우 `preprocess.py`로 생성한 `data_stage1.csv`의 SHA-256이 `SHA256SUMS`와 일치하는지 확인하세요.
