"""저장된 calibrated XGBoost 모델의 개별공정성(individual fairness)을 평가합니다.

재학습 없이 `models/` 의 artifact와 `data/processed/` 의 CSV만 사용합니다.
결과는 JSON으로 저장합니다.

지표: 유사 환자 예측 일관성 (local consistency)
    개별공정성은 "유사한 사람은 유사한 결과를 받아야 한다"는 원칙입니다.
    여기서는 "유사한 환자"를 설문 18문항 중 한 문항의 응답만 Likert 1점 다른 가상의 환자로 정의합니다.

    1. 환자 i의 응답 x_i에 대해 모델 예측 f(x_i)를 구합니다.
    2. 문항 q와 방향 d(-1 또는 +1)마다 x_i의 q번 응답만 d만큼 바꾼 유사 환자 x_i^(q,d)를 만듭니다.
       바꾼 응답이 1~5 범위를 벗어나면 그 쌍은 만들지 않습니다.
    3. 유사 환자 쌍 중 f(x_i) == f(x_i^(q,d)) 인 쌍의 비율을 계산합니다.

        consistency = (예측이 일치한 유사 환자 쌍의 수) / (전체 유사 환자 쌍의 수)

실행 예시
    python src/smdart/fairness_eval.py                 # test split
    FAIRNESS_SPLIT=train python src/smdart/fairness_eval.py
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from causal_intervention import CausalInterventionClassifier  # noqa: F401 (저장된 모델 로드에 필요)

# 경로 설정: 환경변수가 없으면 프로젝트 내부 기본 경로를 사용합니다.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
FAIRNESS_OUTPUT_DIR = OUTPUT_DIR / "fairness"

SPLIT = os.getenv("FAIRNESS_SPLIT", "test")  # train | test | all

MODEL_FILE = "calibrated_xgboost_model.pkl"
FEATURE_FILE = "feature_names.pkl"

LIKERT_MIN, LIKERT_MAX = 1, 5


def sanitize_feature_names(cols: pd.Index) -> pd.Index:
    """XGBoost가 허용하지 않는 feature name 문자를 train.py와 동일하게 치환합니다."""
    cols = cols.astype(str)
    cols = cols.str.replace("[", "(", regex=False)
    cols = cols.str.replace("]", ")", regex=False)
    cols = cols.str.replace("<", "less_than", regex=False)
    return cols


def to_model_input(df_raw: pd.DataFrame, feature_names) -> pd.DataFrame:
    """원본 행을 추론 스크립트와 동일한 방식으로 모델 입력 feature로 변환합니다."""
    df = df_raw.copy()
    df.columns = sanitize_feature_names(df.columns)
    for c in feature_names:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median()).fillna(0)


def local_consistency(model, X: pd.DataFrame, base_pred: np.ndarray) -> dict:
    """한 문항의 응답이 1점만 다른 유사 환자와의 예측 일치율을 계산합니다."""
    # 설문 작성 여부 flag처럼 값이 하나뿐인 컬럼은 Likert 문항이 아니므로 제외합니다.
    likert_cols = [
        c for c in X.columns
        if X[c].nunique() > 1 and X[c].between(LIKERT_MIN, LIKERT_MAX).all()
    ]

    # 모든 유사 환자(문항 x 방향)를 한 번에 만들어 한 번의 예측으로 처리합니다.
    blocks = []
    for col in likert_cols:
        for delta in (-1, 1):
            shifted = X[col] + delta
            valid = shifted.between(LIKERT_MIN, LIKERT_MAX).to_numpy()
            if not valid.any():
                continue
            X_similar = X.loc[valid].copy()
            X_similar[col] = shifted[valid]
            blocks.append((delta, np.flatnonzero(valid), X_similar))

    n_pairs = 0
    n_same = 0
    per_direction = {-1: [0, 0], 1: [0, 0]}  # delta -> [일치 쌍 수, 전체 쌍 수]
    if blocks:
        # 원래 환자의 index를 유지합니다 (같은 환자에서 파생된 행은 같은 index를 가짐).
        similar_pred = np.asarray(model.predict(pd.concat([blk[2] for blk in blocks])))
        offset = 0
        for delta, rows, X_similar in blocks:
            pairs = len(rows)
            matched = int((similar_pred[offset:offset + pairs] == np.asarray(base_pred)[rows]).sum())
            offset += pairs
            n_pairs += pairs
            n_same += matched
            per_direction[delta][0] += matched
            per_direction[delta][1] += pairs

    return {
        "consistency": n_same / n_pairs if n_pairs else float("nan"),
        "n_similar_pairs": n_pairs,
        "n_consistent_pairs": n_same,
        "n_likert_items": len(likert_cols),
        "per_direction": {
            ("minus_1" if d < 0 else "plus_1"): {
                "n_similar_pairs": v[1],
                "n_consistent_pairs": v[0],
                "consistency": v[0] / v[1] if v[1] else float("nan"),
            }
            for d, v in per_direction.items()
        },
    }


def load_split(split: str) -> pd.DataFrame:
    if split == "all":
        return pd.concat([load_split("train"), load_split("test")], ignore_index=True)
    return pd.read_csv(DATA_DIR / f"df_merge_{split}_diagnosis_18_eng.csv")


def main() -> None:
    FAIRNESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_DIR / MODEL_FILE)
    feature_names = joblib.load(MODEL_DIR / FEATURE_FILE)

    df_raw = load_split(SPLIT)
    X = to_model_input(df_raw, feature_names)
    pred = model.predict(X).astype(int)

    summary = {"split": SPLIT, "n": int(len(X)), "n_features": len(feature_names)}
    if "labels" in df_raw.columns:
        y = pd.to_numeric(df_raw["labels"], errors="coerce").astype(int)
        summary["accuracy"] = float(accuracy_score(y, pred))

    summary["individual_fairness"] = local_consistency(model, X, pred)

    out_path = FAIRNESS_OUTPUT_DIR / f"individual_fairness_{SPLIT}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lc = summary["individual_fairness"]
    print("=" * 80)
    print(f"Individual Fairness Evaluation ({SPLIT}, n={summary['n']})")
    print("=" * 80)
    if "accuracy" in summary:
        print(f"Accuracy: {summary['accuracy'] * 100:.2f}%")
    print(
        f"Individual fairness (local consistency): {lc['consistency'] * 100:.2f}% "
        f"({lc['n_consistent_pairs']}/{lc['n_similar_pairs']} similar pairs)"
    )
    for name, d in lc["per_direction"].items():
        print(f"  {name}: {d['consistency'] * 100:.2f}% ({d['n_consistent_pairs']}/{d['n_similar_pairs']})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
