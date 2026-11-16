"""학습된 모델의 train/test 사후 평가, subgroup metric, SHAP 분석을 수행합니다."""

import os
import time
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import shap
import matplotlib.pyplot as plt
import seaborn as sns

from causal_intervention import CausalInterventionClassifier  # noqa: F401 (저장된 모델 로드에 필요)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# 경로 설정: 환경변수가 없으면 프로젝트 내부 기본 경로를 사용합니다.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
TRAIN_CSV = DATA_DIR / "df_merge_train_diagnosis_18_eng.csv"
TEST_CSV = DATA_DIR / "df_merge_test_diagnosis_18_eng.csv"

OUT_DIR = OUTPUT_DIR / "posthoc"
SUBGROUP_COLS = ["age>=65", "gender_male", "region_capital"]

SHAP_SAMPLE_SIZE = 1000
SHAP_MAX_DISPLAY = 20
SHAP_ERROR_FOCUSED_MIN_N = 10
RANDOM_STATE = 42


def ensure_dir(path: str) -> None:
    """결과 저장 폴더가 없으면 생성합니다."""
    os.makedirs(path, exist_ok=True)


def sanitize_feature_names(cols):
    """XGBoost가 허용하지 않는 feature name 문자를 학습 때와 동일하게 치환합니다."""
    cols = pd.Index(cols)
    cols = cols.str.replace("[", "(", regex=False)
    cols = cols.str.replace("]", ")", regex=False)
    cols = cols.str.replace("<", "less_than", regex=False)
    return cols


def fill_missing_like_training(X: pd.DataFrame) -> pd.DataFrame:
    """학습 스크립트와 같은 방식으로 결측값을 채웁니다."""
    X2 = X.copy()
    X2 = X2.apply(pd.to_numeric, errors="ignore")

    num_cols = X2.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X2[num_cols] = X2[num_cols].fillna(X2[num_cols].median())

    for c in X2.columns:
        if X2[c].isnull().any():
            mode = X2[c].mode(dropna=True)
            X2[c] = X2[c].fillna(mode.iloc[0] if len(mode) > 0 else 0)

    return X2


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Confusion matrix를 이미지 파일로 저장합니다."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title, fontsize=13)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_metrics_binary(y_true, y_pred, y_proba1):
    """이진 분류 결과에서 주요 성능 지표와 confusion matrix를 계산합니다."""
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_proba1) if len(np.unique(y_true)) == 2 else np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cm": cm,
    }


def subgroup_metrics_table(y_true, y_pred, y_proba1, meta: pd.DataFrame, group_cols, split_name, out_dir):
    """나이/성별/지역 subgroup별 성능 지표와 confusion matrix를 저장합니다."""
    rows = []
    ensure_dir(out_dir)

    for gcol in group_cols:
        if gcol not in meta.columns:
            continue
        values = sorted([v for v in meta[gcol].dropna().unique()])
        for gval in values:
            mask = (meta[gcol] == gval).values
            n = int(mask.sum())
            if n == 0:
                continue

            yt = y_true[mask]
            yp = y_pred[mask]
            yp1 = y_proba1[mask] if y_proba1 is not None else None

            ratio = float(n / len(y_true)) if len(y_true) > 0 else np.nan
            m = compute_metrics_binary(yt, yp, yp1)

            rows.append({
                "group_col": gcol,
                "group_value": int(gval) if isinstance(gval, (int, np.integer)) else gval,
                "n": n,
                "ratio": ratio,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "auc": m["auc"],
                "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
                "note": "",
            })

            cm_path = os.path.join(out_dir, f"subgroup_cm_{split_name}_{gcol}_{gval}.png")
            plot_confusion_matrix(
                m["cm"],
                class_names=["HD(0)", "PD(1)"],
                title=f"{split_name} | {gcol}={gval} | n={n} | acc={m['accuracy']:.4f} | f1={m['macro_f1']:.4f}",
                save_path=cm_path,
            )

    dfm = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"subgroup_metrics_{split_name}.csv")
    dfm.to_csv(csv_path, index=False)
    return dfm


def run_shap_subgroup_analysis(model, base_model, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, out_dir: str):
    """Calibration 전 base XGBoost 모델로 전체 및 subgroup SHAP 분석을 수행합니다 (보호 속성 개입 적용)."""
    ensure_dir(out_dir)
    rng = np.random.default_rng(RANDOM_STATE)

    n = len(X)
    m = min(SHAP_SAMPLE_SIZE, n)
    idx = rng.choice(n, size=m, replace=False)

    Xs = X.iloc[idx].copy()
    ys = np.asarray(y)[idx]
    metas = meta.iloc[idx].copy()

    shap_exp = model.explain(base_model, Xs)
    shap_vals = shap_exp.values  # binary classification에서 shape은 (sample 수, feature 수)

    global_imp = pd.DataFrame({
        "feature": Xs.columns,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    global_imp.to_csv(os.path.join(out_dir, "shap_importance_all.csv"), index=False)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_vals, Xs, show=False, max_display=SHAP_MAX_DISPLAY)
    plt.title("SHAP Summary (All)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_all.png"), dpi=300, bbox_inches="tight")
    plt.close()

    for gcol in SUBGROUP_COLS:
        if gcol not in metas.columns:
            continue
        for gval in sorted(metas[gcol].dropna().unique()):
            mask = (metas[gcol] == gval).values
            if mask.sum() < 10:
                continue

            Xg = Xs.loc[mask]
            shap_g = shap_vals[mask]

            imp_g = pd.DataFrame({
                "feature": Xg.columns,
                "mean_abs_shap": np.abs(shap_g).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            imp_g.to_csv(os.path.join(out_dir, f"shap_importance_{gcol}_{gval}.csv"), index=False)

            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_g, Xg, show=False, max_display=SHAP_MAX_DISPLAY)
            plt.title(f"SHAP Summary ({gcol}={gval})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"shap_summary_{gcol}_{gval}.png"), dpi=300, bbox_inches="tight")
            plt.close()

    proba1 = model.with_estimator(base_model).predict_proba(Xs)[:, 1]
    yhat = (proba1 >= 0.5).astype(int)

    for gcol in SUBGROUP_COLS:
        if gcol not in metas.columns:
            continue
        for gval in sorted(metas[gcol].dropna().unique()):
            gmask = (metas[gcol] == gval).values
            if gmask.sum() < 10:
                continue

            fn_mask = gmask & (ys == 1) & (yhat == 0)
            fp_mask = gmask & (ys == 0) & (yhat == 1)

            for tag, mask in [("FN", fn_mask), ("FP", fp_mask)]:
                if mask.sum() < SHAP_ERROR_FOCUSED_MIN_N:
                    continue

                Xe = Xs.loc[mask]
                shap_e = shap_vals[mask]

                imp_e = pd.DataFrame({
                    "feature": Xe.columns,
                    "mean_abs_shap": np.abs(shap_e).mean(axis=0),
                }).sort_values("mean_abs_shap", ascending=False)
                imp_e.to_csv(os.path.join(out_dir, f"shap_importance_{gcol}_{gval}_{tag}.csv"), index=False)

                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_e, Xe, show=False, max_display=SHAP_MAX_DISPLAY)
                plt.title(f"SHAP Summary ({gcol}={gval}, {tag})")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"shap_summary_{gcol}_{gval}_{tag}.png"), dpi=300, bbox_inches="tight")
                plt.close()

    return global_imp


def main():
    """모델과 train/test 데이터를 로드한 뒤 사후 평가 및 SHAP 분석을 실행합니다."""
    ensure_dir(OUT_DIR)

    # 1. 학습 결과 artifact 로드
    calibrated_path = os.path.join(MODEL_DIR, "calibrated_xgboost_model.pkl")
    feature_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    base_path = os.path.join(MODEL_DIR, "base_xgb_model.pkl")
    info_path = os.path.join(MODEL_DIR, "training_info.pkl")

    calibrated_model = joblib.load(calibrated_path)
    feature_names = joblib.load(feature_path)

    base_model = None
    if os.path.exists(base_path):
        base_model = joblib.load(base_path)

    training_info = {}
    if os.path.exists(info_path):
        try:
            training_info = joblib.load(info_path)
        except Exception:
            training_info = {}

    print("=" * 80)
    print("Loaded artifacts")
    print("=" * 80)
    print(f"calibrated_model: {calibrated_path}")
    print(f"feature_names:    {feature_path} (n={len(feature_names)})")
    print(f"base_model:       {base_path} ({'FOUND' if base_model is not None else 'MISSING'})")

    # 2. train/test 데이터 로드 및 모델 입력 형태로 정렬
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    def prepare_from_df(df):
        """전처리 CSV를 label, 학습 feature, subgroup 메타데이터로 분리합니다."""
        y = pd.to_numeric(df["labels"], errors="coerce").astype(int).values

        meta_cols = [c for c in ["age>=65", "gender_male", "region_capital", "region_missing", "region_norm"] if c in df.columns]
        meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

        X = df.drop(columns=[c for c in ["ParticipantID", "labels"] if c in df.columns], errors="ignore").copy()
        X.columns = sanitize_feature_names(X.columns)

        drop_analysis_cols = [c for c in ["age>=65", "gender_male", "region_capital", "region_missing", "region_norm"] if c in X.columns]
        if len(drop_analysis_cols) > 0:
            X = X.drop(columns=drop_analysis_cols, errors="ignore")

        for c in feature_names:
            if c not in X.columns:
                X[c] = np.nan
        X = X[feature_names]
        X = fill_missing_like_training(X)

        return X, y, meta

    X_train, y_train, meta_train = prepare_from_df(df_train)
    X_test, y_test, meta_test = prepare_from_df(df_test)

    # 3. Calibrated model 기준 train/test 성능 계산
    yhat_tr = calibrated_model.predict(X_train)
    proba_tr = calibrated_model.predict_proba(X_train)[:, 1]
    met_tr = compute_metrics_binary(y_train, yhat_tr, proba_tr)

    yhat_te = calibrated_model.predict(X_test)
    proba_te = calibrated_model.predict_proba(X_test)[:, 1]
    met_te = compute_metrics_binary(y_test, yhat_te, proba_te)

    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Train acc={met_tr['accuracy']:.4f} f1={met_tr['macro_f1']:.4f} auc={met_tr['auc']:.4f}")
    print(f"Test  acc={met_te['accuracy']:.4f} f1={met_te['macro_f1']:.4f} auc={met_te['auc']:.4f}")

    # 4. Subgroup별 metric 및 confusion matrix 저장
    subgroup_dir = os.path.join(OUT_DIR, "subgroup")
    ensure_dir(subgroup_dir)

    df_sub_tr = subgroup_metrics_table(y_train, yhat_tr, proba_tr, meta_train, SUBGROUP_COLS, "train", subgroup_dir)
    df_sub_te = subgroup_metrics_table(y_test, yhat_te, proba_te, meta_test, SUBGROUP_COLS, "test", subgroup_dir)

    train_step = 1.0 / len(y_train) if len(y_train) > 0 else np.nan
    test_step = 1.0 / len(y_test) if len(y_test) > 0 else np.nan

    print("\n" + "=" * 80)
    print("Accuracy sensitivity")
    print("=" * 80)
    print(f"Train n={len(y_train)} -> 1 error changes accuracy by {train_step:.6f} (~{train_step*100:.3f}%)")
    print(f"Test  n={len(y_test)}  -> 1 error changes accuracy by {test_step:.6f} (~{test_step*100:.3f}%)")

    # 5. 분석 summary 저장
    summary = {
        "train": {k: v for k, v in met_tr.items() if k != "cm"},
        "test": {k: v for k, v in met_te.items() if k != "cm"},
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "accuracy_step_train": float(train_step),
        "accuracy_step_test": float(test_step),
        "subgroup_train_csv": os.path.join(subgroup_dir, "subgroup_metrics_train.csv"),
        "subgroup_test_csv": os.path.join(subgroup_dir, "subgroup_metrics_test.csv"),
        "has_base_model_for_shap": bool(base_model is not None),
    }
    with open(os.path.join(OUT_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if base_model is None:
        print("\n" + "=" * 80)
        print("SHAP skipped")
        print("=" * 80)
        print("base_xgb_model.pkl is missing. SHAP should be computed on the base XGBoost model.")
        print("If you still have the training script outputs, re-save the base model once, then rerun this script.")
        return

    # 6. Base XGBoost 모델 기준 SHAP subgroup 분석
    print("\n" + "=" * 80)
    print("Running SHAP subgroup analysis")
    print("=" * 80)

    shap_dir = os.path.join(OUT_DIR, "shap_subgroup")
    t0 = time.time()
    global_imp = run_shap_subgroup_analysis(calibrated_model, base_model, X_train, y_train, meta_train, shap_dir)
    dt = time.time() - t0
    print(f"SHAP done. Time: {dt:.2f}s")
    print("Top 15 mean|SHAP|:")
    print(global_imp.head(15))

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
