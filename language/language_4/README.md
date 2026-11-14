# kobbq-steer
KoBBQ(한국어 편향 질의응답 벤치마크)에서 `Qwen2.5-7B-Instruct` 모델의 편향을 줄이면서, 일반 지식 성능(KMMLU)은 유지하는 프로젝트입니다.
별도의 모델 학습 없이, 추론 시점에만 모델 내부 활성값을 조정하는 방식으로 편향을 완화합니다.

**Method.** 
1. 모델의 잔차 스트림(Residual Stream)에서 **고정관념에 따른 응답과 '알 수 없음' 응답 간의 방향 벡터 r**을 추출
2. 이때 CAA(Contrastive Activation Addition)의 평균 차이(Difference-in-Means) 방식을 사용
3. 추론 과정에서는 모델의 16번째 레이어에 다음 수식을 적용:  `h ← h + λ·(wᵀh)·r`. 여기서 w는 일반적인 한국어 프롬프트의 영공간(Null Space) 내에서 닫힌 형태의 해(Closed-form Solution)로 계산한 게이트 벡터. (AlphaSteer(Sheng et al., arXiv 2506.07022) 방법을 기반)
4. 핵심: 편향과 관련된 입력에 대해서만 모델의 활성값을 조정하고, KMMLU와 같은 일반 지식 문제에서는 기존 모델의 동작을 최대한 유지.

**Result** (KoBBQ test 13,376 / KMMLU test 35,030, evaluated once):

| | KoBBQ ambig acc | KoBBQ ambig bias | KMMLU acc |
|---|---|---|---|
| base | 88.19% | 0.095 | 49.42% |
| steered | 91.43% | 0.067 | 49.43% (relative drop −0.01%) |

## Setup

실험 환경: Python 3.10, PyTorch 2.6.0 (CUDA 12.4), transformers 4.57.6, one NVIDIA RTX A6000 48 GB.

환경 설치:
```bash
conda create -n ai_dep_lang python=3.10 -y && conda activate ai_dep_lang
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.6 accelerate==1.15.0 datasets==5.0.1 numpy==2.2.6 pandas==2.3.3 \
            pyarrow==25.0.1 scikit-learn==1.7.2 matplotlib==3.10.9 pyyaml==6.0.3 pytest==9.1.1 \
            "huggingface_hub[cli]==0.36.2"

git clone https://github.com/naver-ai/KoBBQ data/ext/KoBBQ     # data + official evaluation code (tested at 0dd165a)
huggingface-cli download HAERAE-HUB/KMMLU --repo-type dataset --include "data/*-dev.csv" "data/*-test.csv" \
    --local-dir data/raw/HAERAE-HUB__KMMLU
```

모델 파일(약 15GB)은 최초 실행 시 `artifacts/hf_cache`에 자동으로 다운로드됩니다. 모든 경로 설정은 `configs/default.yaml`에서 관리합니다.

## 제공된 Steering 파라미터 평가

이미 저장된 Steering 파라미터를 사용해 결과를 재현할 수 있음. 

```bash
python scripts/01_prepare_data.py --eval_only     # data/processed/{kobbq,kmmlu_dev}.parquet
CUDA_VISIBLE_DEVICES=0 python scripts/08_final_eval.py
cat results/final_table.csv
```

총 소요 시간: 2.5 h on one A6000.

- GPU가 2개 있다면 `--conditions base`와 `--conditions alphasteer`를 각각 다른 GPU에서 실행해 병렬로 평가할 수 있습니다.
- 이후에 parameter 없이 평가 script를 한 번 더 실행하면 결과표가 생성됨.
- 정확도와 편향 점수는 KoBBQ 공식 평가 코드와 교차 검증 완료.

## Rebuild the steering parameter

전체 파이프라인을 실행하려면 다음 명령어를 사용: `scripts/run_all.sh` (about 9.5 h on 2 GPUs)

단계: 
1. data preparation (01; also needs `nlpai-lab/kullm-v2`, `kyujinpy/KOpen-platypus`, `kuotient/gsm8k-ko`, `CarrotAI/ko-code-alpaca-QA` and the KMMLU train split under `data/raw/<owner>__<name>/`)
2. activation extraction (02)
3. baseline (03)
4. directions (04)
5. null-space steering fit (05)
6. layer candidates (06)
7. validation sweep and selection (07)
8. test evaluation (08). Selection uses validation data only.

## Layout

```
configs/default.yaml   all settings
src/                   data loading, prompts, scoring, extraction, directions, AlphaSteer solver, hooks, evaluation
scripts/               pipeline steps 01–08, run_all.sh
steering/              selected steering parameter (layer 16, p = 0.3, α = 100, r = CAA direction)
results/qwen2.5-7b/    selected_alphasteer.json (λ = 0.1 and the settings above)
splits/                KoBBQ template-level train/val/test split
tests/                 null-space leakage, closed form vs. pinv, λ = 0 bit identity
```

## References
- Sheng et al., *AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint*, arXiv 2506.07022
- Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*, ACL 2024
- Jin et al., *KoBBQ: Korean Bias Benchmark for Question Answering*, TACL 2024
- Son et al., *KMMLU: Measuring Massive Multitask Language Understanding in Korean*, arXiv 2402.11548
