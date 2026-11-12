# 공인시험 재현 환경 (env.md)

측정을 실제로 수행한 서버 기준. 날짜: 2026-11-10.

## 1. 하드웨어 / OS

| 항목 | 값 |
|---|---|
| 호스트 | cherry.kaist.ac.kr (eno1 IP 143.x.x.x) |
| CPU | Intel Xeon Gold 6326 @ 2.90GHz · 2 socket × 16 core = 32C/64T |
| RAM | 377 GB |
| GPU | **4× NVIDIA RTX A6000 (각 48GB / 49140 MiB)** · Driver 535.183.01 · CUDA 12.2 |
| 저장소 | `/data1` 51 TB (ext, 사용 29T/51T) |
| OS | Ubuntu 20.04.6 LTS (Focal) |

- 측정에 사용한 GPU: 빈 GPU를 `nvidia-smi`로 확인 후 **GPU 0(신용3 응답시간/정확도), GPU 1(신용1)**
  사용. (GPU 2·3은 타 사용자 점유 시간대 있었음 — 응답시간 측정은 전용 GPU에서 수행.)

## 2. 소프트웨어 (conda env `rule_edit`, python 3.10.22)

requirements.txt 그대로 설치 (A6000=sm_86이라 torch 2.4.1 핀 버전 그대로 CUDA 동작):

| 패키지 | 버전 |
|---|---|
| torch | **2.4.1+cu121** (cuda available=True) |
| transformers | 4.40.2 |
| tokenizers | 0.19.1 |
| numpy | 1.24.4 |
| pandas | 2.0.3 |
| scikit-learn | 1.3.2 |
| sentencepiece | 0.2.0 |
| accelerate | 1.0.1 |
| safetensors | 0.5.3 |
| (추가) matplotlib | 3.10.9 — locate/edit 스크립트 import용 |

- 전체 패키지 목록: [`_conda_list_export.txt`](./_conda_list_export.txt) (`conda list -n rule_edit --export`, 107줄).

## 3. 데이터 / 모델 / 편집설정 SHA-256

### 시험 데이터 (test.csv, 평가 전용)
| 파일 | 행수 | SHA-256 |
|---|---|---|
| Regulation/data/test/text/test.csv | 38,232 (dropna 후 38,160 평가) | `5e630e6a8dbb13a9299693b74d4aacf7295368e30d160fb73f8208621141c297` |
| Bias/data/test/text/test.csv | 16,281 | `cfdb68c36ee5237eec7e369f6a22db72a70e05d5c55dd6b41045dfd5c5efdbad` |

### 모델 (model.safetensors)
| 모델 | SHA-256 |
|---|---|
| Bias cgmix_rel (`Bias/model/finetuned_xlm-roberta-large_cgmix_rel/best`) | `ac2afefb3f5f08a1f2bb37f353edb2b631a72d9aba57d7724de2c5b6aa83f1b5` |
| Regulation (`editing_centroid/xlm-roberta-large-rule`, HF yongukpark/xlm-roberta-large-rule와 동일) | `bbe823e8d6fdb7b66dc9088f242af9910ee6a0c4bd8ff98b30a8f62ec679846e` |

- Bias 모델은 공개 Google Drive에서 받은 원본 `cgmix_rel.tar.gz`
  SHA-256 `583cd4056982ec110234f0b008b606652f1f45e41186bc6bc3ea5febed6a9feb` 를 전개한 것.

### 편집 설정 (shift_config.pt, 재생성본 — 기존 report와 수치 일치 확인)
| 파일 | SHA-256 |
|---|---|
| Bias layer9_seed0 | `82e23bfe51256b7ef7b482ea133cdde21ca259a7cccb2b05d6556701cf5bf1c3` |
| Bias layer9_seed1 | `3910a6bc62e8ab9a7ad9d123698a3c669afea93fd14649436b26260f9cc4a1f8` |
| Bias layer9_seed2 | `c94d9e6bbcf1517f1824cd3e9b1cfa539932bdd5efe8e4e0908cea523ece44fe` |
| Bias layer9_seed3 | `5b9e54c346e392371e762cf5830f9a69d7951327d78aa9652bbd43b37778af2d` |
| Regulation rule1 v3b2 matched | `3f8e865ed13fee02367017b6cb36c9724c974335b3b8eb31ad449d7dc4ab29d5` |

(.pt는 매 저장 시 메타가 달라 SHA가 바뀔 수 있음 — 내용 동등성은 delta_v_norm/best_alpha 수치 대조로 검증. 작업1 참조.)

## 4. 재현 명령어 (정본)

```bash
# 공통
source /data1/jinsil/miniconda3/etc/profile.d/conda.sh && conda activate rule_edit
cd /data1/jinsil/aidep/AI-DEP_RuleEditing-main

# [신용3] 정확도·재현율·응답시간
CUDA_VISIBLE_DEVICES=0 python cert/cert_credit3_regulation.py \
    --model_dir /data1/jinsil/aidep/editing_centroid/xlm-roberta-large-rule \
    --shift_config Regulation/centroid/result/centroid_matched/edit_matched_v3b2/rule1_matched_shift_config.pt \
    --latency_n 1000 --save_json cert/results/credit3_v3b2_run1.json
# (응답시간 전체 1회: 위에서 --skip_accuracy --latency_n -1)

# [신용1] 정확도·공정성(DEO), seed0~3
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 python cert/cert_credit1_bias.py \
      --shift_config Bias/result/bias_centroid/edit/layer9_seed$s/shift_config.pt \
      --alpha 0.9 --with_base --save_json cert/results/credit1_alpha0.9_seed$s.json
done
```

### 편집 설정 재생성(작업1, 참고)
```bash
# Regulation: locate(캐시 재생성) → matched edit
cd Regulation/centroid/src/editing
CUDA_VISIBLE_DEVICES=0 python centroid_locate_synth.py --rule 1 --context_mode v3b2 \
    --n_pairs 500 --n_patch 100 --seed 0 \
    --model_name /data1/jinsil/aidep/editing_centroid/xlm-roberta-large-rule --overwrite
CUDA_VISIBLE_DEVICES=0 python centroid_edit_matched.py --version v3b2 --rule 1 \
    --alphas 0.5 1.0 2.0 --n_cal_extract 1000 --n_cal 150 --max_safe_false_flip 0.02 --seed 0 \
    --model_name /data1/jinsil/aidep/editing_centroid/xlm-roberta-large-rule --overwrite
# Bias: 4 seed (import 경로 fix: PYTHONPATH로 Regulation/centroid의 probe_weight_raw_space 사용)
cd Bias/src/editing
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../Regulation/centroid:../../../Regulation/centroid/src/editing \
  python bias_centroid_edit.py --layer 9 --seed $s \
      --alphas 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.1 1.25 --gpu_id 0
done
```
