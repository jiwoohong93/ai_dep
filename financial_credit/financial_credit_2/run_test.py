"""ALINCredit 공인시험 실행 스크립트.

    python run_test.py --item 1    # 시험항목 1. 공정성 DEO x100
    python run_test.py --item 2    # 시험항목 2. 정책 반영 성능 평균 하락률
    python run_test.py             # 두 항목 모두

1) 시험 데이터 8개 파일의 SHA-256 해시를 출력하고 기준값과 비교한다.
2) 학습된 ALINCredit 모델(checkpoints/alincredit_seed1.joblib)을 불러와 테스트 분할(seed 1, 70/15/15)을 평가한다.
   (python run_test.py --train : 모델을 다시 학습하여 checkpoints/ 에 저장한 뒤 평가)
3) 선택한 시험항목의 값을 출력하고 기준 충족 여부를 판정한다.
     - [금융/신용2] 공정성 DEO x100 = |TPR(남) - TPR(여)| x 100        (기준: 5 미만)
     - [금융/신용2] 정책 반영 성능 평균 하락률 = (Acc_정책미반영 - Acc_정책반영) / Acc_정책미반영 x 100 (기준: 3 % 미만)
결과는 results/test_result_item1.json, test_result_item2.json (두 항목 모두: test_result.json) 에도 저장된다.
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)  # aif360 optional-module notices
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

DEO_LIMIT = 5.0
DROP_LIMIT = 3.0

# 시험 데이터 기준 해시 (SHA-256)
EXPECTED_SHA256 = {
    "German": "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871",
    "LendingClub": "5f58997fd1d8bcf5be86a8d8b1d0144d0266296c567949f325970f6fe9312b10",
    "Taiwan": "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933",
    "Prosper": "71c1c1931f65bc4d2e3a8f3603d865e074e327b2271590f3a06829197b9cce0f",
    "HomeCredit": "8888c658ea7a12f664a958b8f11096d574eb450bb773934bbc003ddf12807656",
    "HMEQ": "ec24c35e24306e045b86bc55a00b099623707e042b9d4adbf36f5edfb03dba44",
    "CreditApproval": "89b44bbe837e387db12f3998e74b88c2dc35458c615c80c7bcc4e7764e30fcbe",
    "Australian": "f17305ac6b6709d59e2c741b866555485d93b175b105544d1d1f44753ac16a94",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--item", type=int, choices=[1, 2], default=None,
                    help="1: 공정성 DEO x100, 2: 정책 반영 성능 평균 하락률 (생략 시 두 항목 모두)")
    ap.add_argument("--train", action="store_true", help="모델을 다시 학습하고 checkpoints/ 에 저장")
    args = ap.parse_args()
    title = {None: "공정성, 정책 반영 성능 평균 하락률", 1: "시험항목 1. 공정성", 2: "시험항목 2. 정책 반영 성능 평균 하락률"}[args.item]
    ckpt = os.path.join(ROOT, "checkpoints", f"alincredit_seed{args.seed}.joblib")
    train = args.train or not os.path.exists(ckpt)

    from evaluate import sha256
    from algorithms.inprocessing.alincredit_function import DATA_FILES, DATA_DIR, load_raw

    line = "=" * 70
    print(line)
    print(f" ALINCredit V1.0 공인시험  |  [금융/신용2] {title}")
    print(line)

    print(f"[1/3] 시험 데이터 확인 (경로: {os.path.abspath(DATA_DIR)})")
    hash_ok = True
    hashes = {}
    for name, fn in DATA_FILES.items():
        h = sha256(os.path.join(DATA_DIR, fn))
        hashes[name] = h
        exp = EXPECTED_SHA256.get(name)
        ok = exp == h
        hash_ok &= ok
        print(f"  {name:<16}{fn:<38}{h}  {'OK' if ok else '불일치'}")
    print(f"  -> 데이터 해시 {'일치' if hash_ok else '불일치'}")

    t0 = time.time()
    from algorithms.inprocessing.ALINCredit import ALINCredit
    if train:
        print(f"\n[2/3] 모델 학습 및 평가 (seed {args.seed}) ...", flush=True)
        model = ALINCredit(dataset_name="credit_union", seed=args.seed, raw=load_raw())
        model.run()
        model.save_models(ckpt)
        print(f"  학습 모델 저장: {ckpt}")
    else:
        print(f"\n[2/3] 학습된 모델로 테스트 데이터 평가 (seed {args.seed}) ...", flush=True)
        print(f"  모델 파일: {ckpt}")
        print(f"  SHA-256 : {sha256(ckpt)}")
        model = ALINCredit(dataset_name="credit_union", seed=args.seed, raw=load_raw())
        model.run_inference(ckpt)
    s = model.summary
    print(f"  완료 ({time.time() - t0:.0f} 초)")

    deo, drop = s["deo_x100"], s["policy_drop"]
    acc_fair, acc_ref = s["pooled_acc"], s["pooled_acc_unconstrained"]
    deo_pass, drop_pass = deo < DEO_LIMIT, drop < DROP_LIMIT

    print(f"\n[3/3] 시험 결과")
    out = {"seed": args.seed, "mode": "train" if train else "inference",
           "data_sha256": hashes, "data_hash_ok": hash_ok}
    if args.item in (None, 1):
        print(f"  DEO 산출 대상: 테스트셋 중 성별 정보가 있는 데이터 전체")
        print(line)
        print(f"  시험항목 1. 공정성 DEO x100 = {deo:.2f}   (기준 {DEO_LIMIT:g} 미만)   "
              f"{'충족' if deo_pass else '미충족'}")
        print(line)
        out.update({"deo_x100": round(deo, 4), "deo_pass": deo_pass})
    if args.item in (None, 2):
        print(f"  테스트셋 전체 정확도: 정책 미반영 {acc_ref:.2f} %  /  정책 반영 {acc_fair:.2f} %")
        print(line)
        print(f"  시험항목 2. 정책 반영 성능 평균 하락률 = {drop:.2f} %   (기준 {DROP_LIMIT:g} % 미만)   "
              f"{'충족' if drop_pass else '미충족'}")
        print(line)
        out.update({"policy_drop_pct": round(drop, 4), "acc_policy_applied": round(acc_fair, 4),
                    "acc_policy_not_applied": round(acc_ref, 4), "drop_pass": drop_pass})

    name = "test_result.json" if args.item is None else f"test_result_item{args.item}.json"
    path = os.path.join(ROOT, "results", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"결과 파일: {path}")


if __name__ == "__main__":
    main()
