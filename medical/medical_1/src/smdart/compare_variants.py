"""보호 속성 사용 방식에 따른 세 가지 모델을 같은 조건에서 비교합니다.

    with_protected: 설문 응답 + 보호 속성 입력, 개입 없음 (baseline)
    survey_only   : 설문 응답만 입력
    causal        : 설문 응답 + 보호 속성 입력, 추론 시 보호 속성에 인과 개입 (최종 모델)

세 모델 모두 train.py와 같은 XGBoost 설정과 sigmoid 확률 보정을 사용합니다.

실행 예시
    python src/smdart/compare_variants.py                        # 고정 train / test 분할
    COMPARE_CV_REPEATS=10 python src/smdart/compare_variants.py  # + 전체 데이터 repeated 5-fold CV
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from causal_intervention import CausalInterventionClassifier, PROTECTED_FEATURES, protected_distribution
from fairness_eval import local_consistency, sanitize_feature_names

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
COMPARE_OUTPUT_DIR = OUTPUT_DIR / "comparison"

CV_REPEATS = int(os.getenv("COMPARE_CV_REPEATS", "0"))
DEVICE = os.getenv("XGB_DEVICE", "cpu")
RANDOM_STATE = 42

NON_FEATURE_COLS = ["ParticipantID", "labels", "region_missing", "region_norm"] + PROTECTED_FEATURES
VARIANTS = ["with_protected", "survey_only", "causal"]
METRICS = ["accuracy", "macro_f1", "auc", "individual_fairness", "protected_invariance"]
GAP_METRICS = [f"prediction_gap_{c}" for c in PROTECTED_FEATURES]


def load_split(split: str):
    df = pd.read_csv(DATA_DIR / f"df_merge_{split}_diagnosis_18_eng.csv")
    df.columns = sanitize_feature_names(df.columns)
    survey_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    Q = df[survey_cols].apply(pd.to_numeric, errors="coerce")
    Q = Q.fillna(Q.median()).fillna(0)
    A = df[PROTECTED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    y = pd.to_numeric(df["labels"], errors="coerce").astype(int).to_numpy()
    return Q, A, y


def make_estimator():
    base = xgb.XGBClassifier(
        objective="binary:logistic", max_depth=5, learning_rate=0.1, n_estimators=200,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss",
        tree_method="hist", device=DEVICE,
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=5, ensemble=True)


class SurveyOnly:
    """설문 응답만 입력받는 모델."""

    def __init__(self, estimator):
        self.estimator = estimator

    def predict_proba(self, Q, A):
        return self.estimator.predict_proba(Q)[:, 1]


class WithProtected:
    """보호 속성을 그대로 입력받는 모델 (개입 없음)."""

    def __init__(self, causal_model):
        self.causal_model = causal_model

    def predict_proba(self, Q, A):
        return self.causal_model.predict_proba_observed(Q, A)[:, 1]


class Causal:
    """보호 속성에 인과 개입을 적용한 모델."""

    def __init__(self, causal_model):
        self.causal_model = causal_model

    def predict_proba(self, Q, A):
        return self.causal_model.predict_proba(Q)[:, 1]


def fit_variants(Q, A, y):
    survey_estimator = make_estimator().fit(Q, y)
    full_estimator = make_estimator().fit(pd.concat([Q, A], axis=1), y)
    combinations, weights = protected_distribution(A)
    causal_model = CausalInterventionClassifier(full_estimator, Q.columns.tolist(), PROTECTED_FEATURES, combinations, weights)
    return {
        "survey_only": SurveyOnly(survey_estimator),
        "with_protected": WithProtected(causal_model),
        "causal": Causal(causal_model),
    }, combinations


class _BatchAdapter:
    """local_consistency()가 요구하는 predict(X) 형태로 맞추는 어댑터.

    유사 환자는 원래 환자의 보호 속성을 그대로 가지므로, 입력 행의 원래 index로 보호 속성을 찾습니다.
    """

    def __init__(self, variant, A):
        self.variant = variant
        self.A = A

    def predict(self, Q):
        return (self.variant.predict_proba(Q, self.A.loc[Q.index]) >= 0.5).astype(int)


def protected_invariance(variant, Q, A, base_pred, combinations):
    """보호 속성만 다른 값으로 바꿨을 때 예측이 유지되는 비율."""
    n_pairs = n_same = 0
    observed = A.to_numpy()
    for combo in combinations:
        differs = ~(observed == combo).all(axis=1)
        A_changed = pd.DataFrame(np.tile(combo, (len(Q), 1)), columns=PROTECTED_FEATURES, index=A.index)
        pred = (variant.predict_proba(Q, A_changed) >= 0.5).astype(int)
        n_pairs += int(differs.sum())
        n_same += int(((pred == base_pred) & differs).sum())
    return n_same / n_pairs if n_pairs else float("nan")


def prediction_gaps(pred, A) -> dict:
    """보호 속성 집단 간 복막투석 예측 비율 차이 (속성=1 집단 - 속성=0 집단)."""
    gaps = {}
    for col in PROTECTED_FEATURES:
        g1 = (A[col] == 1).to_numpy()
        g0 = (A[col] == 0).to_numpy()
        gaps[f"prediction_gap_{col}"] = float(pred[g1].mean() - pred[g0].mean()) if g1.any() and g0.any() else float("nan")
    return gaps


def evaluate(variant, Q, A, y, combinations):
    Q = Q.reset_index(drop=True)
    A = A.reset_index(drop=True)
    proba = variant.predict_proba(Q, A)
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "auc": roc_auc_score(y, proba),
        "individual_fairness": local_consistency(_BatchAdapter(variant, A), Q, pred)["consistency"],
        "protected_invariance": protected_invariance(variant, Q, A, pred, combinations),
    }
    return metrics, pred


def run_fixed_split() -> pd.DataFrame:
    Q_train, A_train, y_train = load_split("train")
    Q_test, A_test, y_test = load_split("test")
    variants, combinations = fit_variants(Q_train, A_train, y_train)
    rows = []
    for name in VARIANTS:
        metrics, pred = evaluate(variants[name], Q_test, A_test, y_test, combinations)
        rows.append({"variant": name, **metrics, **prediction_gaps(pred, A_test.reset_index(drop=True))})
    return pd.DataFrame(rows)


def run_cross_validation(repeats: int):
    """Repeated 5-fold CV. fold별 지표와, 반복마다 out-of-fold 예측 전체로 계산한 집단 간 예측 차이를 반환합니다."""
    Q, A, y = load_split("all")
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=RANDOM_STATE)
    fold_records, gap_records = [], []
    oof = {name: np.zeros(len(y), dtype=int) for name in VARIANTS}
    for fold, (train_idx, test_idx) in enumerate(splitter.split(Q, y), start=1):
        variants, combinations = fit_variants(Q.iloc[train_idx], A.iloc[train_idx], y[train_idx])
        for name in VARIANTS:
            metrics, pred = evaluate(variants[name], Q.iloc[test_idx], A.iloc[test_idx], y[test_idx], combinations)
            oof[name][test_idx] = pred
            fold_records.append({"fold": fold, "variant": name, **metrics})
        print(f"CV fold {fold}/{5 * repeats} done", flush=True)
        if fold % 5 == 0:  # 한 번의 반복이 끝나면 전체 환자의 out-of-fold 예측이 채워집니다.
            for name in VARIANTS:
                gap_records.append({"repeat": fold // 5, "variant": name, **prediction_gaps(oof[name], A)})
    return pd.DataFrame(fold_records), pd.DataFrame(gap_records)


def print_table(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    COMPARE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed = run_fixed_split()
    fixed.to_csv(COMPARE_OUTPUT_DIR / "variant_comparison_test.csv", index=False)
    print_table(fixed, "Variant comparison (fixed train / test split)")

    if CV_REPEATS > 0:
        folds, gaps = run_cross_validation(CV_REPEATS)
        folds.to_csv(COMPARE_OUTPUT_DIR / "variant_comparison_cv_folds.csv", index=False)
        summary = pd.concat(
            [
                folds.groupby("variant")[METRICS].agg(["mean", "std"]).loc[VARIANTS],
                gaps.groupby("variant")[GAP_METRICS].agg(["mean", "std"]).loc[VARIANTS],
            ],
            axis=1,
        )
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(COMPARE_OUTPUT_DIR / "variant_comparison_cv.csv", index=False)
        print_table(summary, f"Variant comparison (repeated 5-fold CV x {CV_REPEATS})")

    print(f"\nSaved: {COMPARE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
