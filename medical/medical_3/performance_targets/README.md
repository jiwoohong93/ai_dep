# 협약 성과지표 재현 — 예측정확도 · 공정성 · 정책반영 성능하락률

과제 협약 성과지표를 측정한 학습·평가 코드와 결과입니다. 상위 폴더의 ② 실행 파일(`algorithms/preprocessing/SubgroupBalancedResampling.py`)이 이 코드를 그대로 호출합니다.

| 성과지표 | 목표 | 정의 |
|---|---|---|
| 예측정확도 | ≥ 94% | 테스트 438장 전체 정확도 (argmax@0.5) |
| 공정성 (DEO) | ≤ 5% | \|TPR_여 − TPR_남\| × 100 (원 과제 `train.py`의 `evaluate_DEO` 정의 그대로: TPR은 label==1 클래스 기준) |
| 정책반영 성능평균하락률 | ≤ 3% | (acc_공정성미적용 − acc_공정성적용) / acc_공정성미적용 × 100 |

- **벤치마크**: 원 과제와 동일한 Four-Findings 전문가 판독 라벨(테스트+검증 CSV 병합 4,376장), 이미지 단위 분할
  `train_test_split(test_size=0.10, random_state=42)` → 학습 3,938 / 테스트 438 (테스트 기흉 양성 30장: 여 16 / 남 14).
- **공정성 적용 모델**: (라벨×성별) 그룹 WeightedRandomSampler. **미적용 모델**: 일반 셔플. 두 모델을 같은 레시피로 각각 학습.
- **레시피**: rad-DINO(`microsoft/rad-dino` @ `110cbc18d5133582e320b43d53bf5c44e410c936`) 파인튜닝, 15 epoch, 강한 증강,
  클래스 가중 CE, TTA(좌우반전), lr 2e-5, batch 16.
  ResNet-18(baseline): ImageNet 사전학습, 20 epoch, 강한 증강, 클래스 가중 CE, lr 5e-5, batch 16.

## 결과

| 모델 | 학습 시드 | 예측정확도 | 공정성 DEO | 정책반영 하락률 |
|---|---|---|---|---|
| ResNet-18 (baseline) | 42 | 90.41 | 3.51 | 0.25 |
| rad-DINO + ② | 42 | 94.52 | 0.40 | 0.24 |
| rad-DINO + ② | 1 | 94.52 | 1.46 | −2.48 |
| rad-DINO + ② | 2 | 94.98 | 0.59 | −0.48 |
| rad-DINO + ② | 3 | 92.92 | 3.88 | 1.69 |
| rad-DINO + ② | 4 | 94.75 | 0.79 | −0.24 |
| **rad-DINO + ② (5개 평균 ± 표준편차)** | | **94.34 ± 0.73** | **1.42 ± 1.28** | **−0.25 ± 1.34** |

- 목표: 예측정확도 ≥ 94, 공정성 DEO ≤ 5, 정책반영 하락률 ≤ 3. ResNet-18은 정확도 미달, rad-DINO + ②는 5개 평균으로 3종 달성.
- ② = (라벨 × 성별) 그룹 균형 샘플링(WeightedRandomSampler). 하락률은 같은 backbone의 ② 미적용 모델 대비.
- 테스트 438장이라 1장 = 0.23%p입니다.
- 결과 파일: `results/target_push_resnet18_e20_strong_cw_s42.json`, `results/target_push_raddino_e15_strong_cw_tta_s{42,1,2,3,4}.json`,
  실행 로그 `results/run_raddino_s*.log`.

## 데이터 준비
1. **Four-Findings 라벨 CSV**: `data/`에 동봉 (원 과제 저장소에 포함돼 있던 파일 그대로).
   NIH ChestX-ray14의 전문가 판독 라벨(Google, Majkowska et al., *Radiology* 2020).
2. **NIH ChestX-ray14 영상**(필요한 4,376장만): HuggingFace 공개 미러
   `alkzar90/NIH-Chest-X-ray-dataset`에서 아카이브(images_001~012.zip)를 하나씩 받아 필요한 파일만 풀고 지웁니다.
   총 전송량 약 45GB, 디스크 최대 사용량은 아카이브 1개 + 추출 영상(약 2GB).
   ```bash
   python download_nih_subset.py        # -> data/datasets/nih-chest-xrays/data/versions/3/images_XXX/images/
   ```
   이미 NIH 영상이 있으면 `NIH_ROOT=<.../versions/3>` 로 지정하면 됩니다.

## 실행
```bash
pip install -r ../requirements.txt     # torch·transformers 포함
python target_push.py --backbone raddino  --epochs 15 --aug strong --fix_classweight --tta --lr 2e-5 --batch_size 16 --seed 42
python target_push.py --backbone resnet18 --epochs 20 --aug strong --fix_classweight       --lr 5e-5 --batch_size 16 --seed 42   # baseline
python target_push.py --backbone raddino --tta --smoke   # 데이터·분할·모델 점검만 (학습 없음)
```
GPU 1장(RTX 2080 Ti 11GB) 기준 rad-DINO 1회(② 적용·미적용 2개 모델) 약 35~40분. 출력 마지막 줄 `TARGET CHECK`에 세 지표 판정이 나옵니다.
