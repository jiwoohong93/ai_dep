"""저장된 모델로 test set 추론, 평가, 리포트, SHAP 결과를 생성합니다."""

import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, confusion_matrix, f1_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from causal_intervention import CausalInterventionClassifier  # noqa: F401 (저장된 모델 로드에 필요)


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
INFERENCE_OUTPUT_DIR = OUTPUT_DIR / "inference" / "test"
FIGURE_DIR = INFERENCE_OUTPUT_DIR / "figures"
REPORT_DIR = INFERENCE_OUTPUT_DIR / "reports"
SHAP_OUTPUT_DIR = INFERENCE_OUTPUT_DIR / "shap"
TEST_CSV_NAME = "df_merge_test_diagnosis_18_eng.csv"

MODEL_FILE = "calibrated_xgboost_model.pkl"
BASE_MODEL_FILE = "base_xgb_model.pkl"
FEATURE_FILE = "feature_names.pkl"
TRAININFO_FILE = "training_info.pkl"
SHAP_TOP_N = 15
SHAP_TEST_DIR = "test_samples"

for output_dir in (FIGURE_DIR, REPORT_DIR, SHAP_OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)


# 1. XGBoost CUDA 사용 가능 여부 확인
def check_gpu_availability() -> bool:
    try:
        test_model = xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            device="cuda",
            n_estimators=1,
            max_depth=1,
            eval_metric="logloss",
        )
        X_ = np.random.rand(32, 5)
        y_ = np.random.randint(0, 2, size=(32,))
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            test_model.fit(X_, y_)
        messages = " ".join(str(r.message) for r in records)
        if "No visible GPU" in messages or "not compiled with CUDA" in messages or "device to CPU" in messages:
            return False
        return True
    except Exception:
        return False


GPU_AVAILABLE = check_gpu_availability()
print("=" * 80)
print("Loading model...")
print("=" * 80)
print(f"GPU Available (current env): {GPU_AVAILABLE}")


# 2. 학습된 모델 artifact 로드
model_path = MODEL_DIR / MODEL_FILE
base_model_path = MODEL_DIR / BASE_MODEL_FILE
feature_path = MODEL_DIR / FEATURE_FILE
traininfo_path = MODEL_DIR / TRAININFO_FILE

model = joblib.load(model_path)
base_model = joblib.load(base_model_path) if os.path.exists(base_model_path) else None
feature_names = joblib.load(feature_path)

training_info = {}
if os.path.exists(traininfo_path):
    try:
        training_info = joblib.load(traininfo_path)
        print(f"Training info loaded. gpu_used (logged): {training_info.get('gpu_used', 'Unknown')}")
    except Exception:
        print("Training info exists but failed to load. Proceeding without it.")
else:
    print("Training info not found. Proceeding without it.")

print(f"\nModel loaded: {model_path}")
print(f"Base model loaded: {base_model_path if base_model is not None else 'Not found'}")
print(f"Required features: {len(feature_names)}")
print("=" * 80)


# 3. 이진 분류 라벨 정의
label_name_map = {
    0: "Hemodialysis (0)",
    1: "Peritoneal dialysis (1)",
}
ordered_labels = [0, 1]
target_names_for_report = [label_name_map[k] for k in ordered_labels]


# 4. 학습 때와 동일한 결측값 처리
def fill_missing_like_training(X: pd.DataFrame) -> pd.DataFrame:
    """학습 스크립트와 같은 방식으로 결측값을 채웁니다.

    숫자형 컬럼은 median, 나머지 컬럼은 mode로 채우며 mode가 없으면 0을 사용합니다.
    """
    X = X.copy()
    X = X.apply(pd.to_numeric, errors="ignore")

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    for c in X.columns:
        if X[c].isnull().any():
            mode = X[c].mode(dropna=True)
            X[c] = X[c].fillna(mode.iloc[0] if len(mode) > 0 else 0)

    return X


def sanitize_feature_names(cols: pd.Index) -> pd.Index:
    """XGBoost가 허용하지 않는 feature name 문자를 train.py와 동일하게 치환합니다."""
    cols = cols.astype(str)
    cols = cols.str.replace("[", "(", regex=False)
    cols = cols.str.replace("]", ")", regex=False)
    cols = cols.str.replace("<", "less_than", regex=False)
    return cols

# 5. Test set 전체 샘플별 SHAP 저장
def run_all_test_sample_shap_analysis(
    model,
    base_model,
    X_full: pd.DataFrame,
    *,
    output_dir: str,
) -> None:
    """Test set의 각 샘플에 대해 SHAP waterfall plot과 기여도 CSV를 저장합니다."""
    if base_model is None:
        print("Base model not found. Skipping all-sample SHAP analysis.")
        return
    if len(X_full) == 0:
        print("No test samples found. Skipping all-sample SHAP analysis.")
        return

    shap_dir = os.path.join(output_dir, SHAP_TEST_DIR)
    os.makedirs(shap_dir, exist_ok=True)

    # 보호 속성에 개입한 예측에 대한 설문 문항별 SHAP 값
    shap_all = model.explain(base_model, X_full)

    print("\n" + "=" * 80)
    print("All Test-Sample SHAP Analysis")
    print("=" * 80)
    print(f"Saving outputs to: {shap_dir}")

    for sample_index in range(len(X_full)):
        sample_df = X_full.iloc[[sample_index]].copy()
        shap_exp = shap_all[sample_index:sample_index + 1]
        shap_values = shap_exp.values[0]

        shap_df = pd.DataFrame({
            "feature": sample_df.columns,
            "feature_value": sample_df.iloc[0].values,
            "shap_value": shap_values,
            "abs_shap_value": np.abs(shap_values),
        }).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

        shap_csv_path = os.path.join(shap_dir, f"shap_single_sample_{sample_index}.csv")
        shap_df.to_csv(shap_csv_path, index=False, encoding="utf-8-sig")

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_exp[0], max_display=SHAP_TOP_N, show=False)
        plt.tight_layout()
        shap_plot_path = os.path.join(shap_dir, f"shap_single_sample_{sample_index}.png")
        plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        if sample_index < 3 or sample_index == len(X_full) - 1:
            print(f"Saved sample {sample_index}: {shap_csv_path}, {shap_plot_path}")

    print(f"Completed SHAP export for {len(X_full)} test samples.")
# 6. Test CSV 로드 및 학습 feature 순서에 맞게 정렬
print("\nTest Dataset Loading and Feature Alignment")
print("-" * 80)

test_file_path = DATA_DIR / TEST_CSV_NAME
df_test_raw = pd.read_csv(test_file_path)
df_test = df_test_raw.copy()

# train.py와 동일하게 XGBoost 비허용 문자를 정리합니다.
df_test.columns = sanitize_feature_names(df_test.columns)

# 정답 라벨을 숫자형으로 변환합니다.
y_test = pd.to_numeric(df_test["labels"], errors="coerce").astype(int)

# 저장된 feature_names.pkl 기준으로 test feature 순서를 강제합니다.
X_test = df_test.copy()

# 누락된 feature는 NaN으로 만든 뒤 학습 때와 같은 방식으로 채웁니다.
for c in feature_names:
    if c not in X_test.columns:
        X_test[c] = np.nan

X_test = X_test[feature_names].copy()
X_test = fill_missing_like_training(X_test)
# 7. Calibrated model의 전체 test set 성능 평가
print("\nFull Test Set Performance Evaluation")
print("-" * 80)

start_time = time.time()
y_pred = model.predict(X_test)
y_pred_proba_all = model.predict_proba(X_test)
y_pred_proba = y_pred_proba_all[:, 1]
total_time = time.time() - start_time

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) == 2 else np.nan
brier = brier_score_loss(y_test, y_pred_proba)
ll = log_loss(y_test, y_pred_proba)

avg_ms = (total_time / len(X_test)) * 1000.0

print(f"\nTotal samples: {len(X_test)}")
print(f"Total inference time: {total_time:.4f} sec")
print(f"Avg time per sample: {avg_ms:.2f} ms")
print(f"Throughput: {len(X_test) / total_time:.1f} samples/sec")
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Macro F1: {f1m:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print(f"Brier score: {brier:.4f}")
print(f"Log loss:    {ll:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=ordered_labels,
    target_names=target_names_for_report,
))

cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names_for_report,
    yticklabels=target_names_for_report,
)
plt.title(
    f"Confusion Matrix - Test Set\nAccuracy: {acc:.4f} ({acc*100:.2f}%) | Macro F1: {f1m:.4f} | AUC: {auc:.4f}",
    fontsize=12,
)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

out_path = FIGURE_DIR / "confusion_matrix_test.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy="quantile")

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Calibrated XGBoost")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive rate")
plt.title(
    f"Calibration Curve - Test Set\nBrier: {brier:.4f} | Log loss: {ll:.4f}",
    fontsize=12,
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

calibration_out_path = FIGURE_DIR / "calibration_curve_test.png"
plt.savefig(calibration_out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {calibration_out_path}")

if base_model is not None:
    print("\nBase XGBoost Test Set Performance Evaluation")
    print("-" * 80)

    base_start_time = time.time()
    # 확률 보정 전 모델에도 같은 개입을 적용해 비교합니다.
    base_intervened = model.with_estimator(base_model)
    y_pred_base = base_intervened.predict(X_test)
    y_pred_base_proba_all = base_intervened.predict_proba(X_test)
    y_pred_base_proba = y_pred_base_proba_all[:, 1]
    base_total_time = time.time() - base_start_time

    acc_base = accuracy_score(y_test, y_pred_base)
    f1m_base = f1_score(y_test, y_pred_base, average="macro")
    auc_base = roc_auc_score(y_test, y_pred_base_proba) if len(np.unique(y_test)) == 2 else np.nan
    brier_base = brier_score_loss(y_test, y_pred_base_proba)
    ll_base = log_loss(y_test, y_pred_base_proba)
    avg_ms_base = (base_total_time / len(X_test)) * 1000.0

    print(f"\nTotal samples: {len(X_test)}")
    print(f"Total inference time: {base_total_time:.4f} sec")
    print(f"Avg time per sample: {avg_ms_base:.2f} ms")
    print(f"Throughput: {len(X_test) / base_total_time:.1f} samples/sec")
    print(f"Accuracy: {acc_base:.4f} ({acc_base*100:.2f}%)")
    print(f"Macro F1: {f1m_base:.4f}")
    print(f"ROC-AUC:  {auc_base:.4f}")
    print(f"Brier score: {brier_base:.4f}")
    print(f"Log loss:    {ll_base:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred_base,
        labels=ordered_labels,
        target_names=target_names_for_report,
    ))

    cm_base = confusion_matrix(y_test, y_pred_base, labels=ordered_labels)
    print("\nConfusion Matrix:")
    print(cm_base)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_base,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names_for_report,
        yticklabels=target_names_for_report,
    )
    plt.title(
        f"Confusion Matrix - Test Set (Base XGBoost)\nAccuracy: {acc_base:.4f} ({acc_base*100:.2f}%) | Macro F1: {f1m_base:.4f} | AUC: {auc_base:.4f}",
        fontsize=12,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    base_cm_path = FIGURE_DIR / "confusion_matrix_test_base.png"
    plt.savefig(base_cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {base_cm_path}")

    prob_true_base, prob_pred_base = calibration_curve(y_test, y_pred_base_proba, n_bins=10, strategy="quantile")

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(prob_pred_base, prob_true_base, marker="o", linewidth=2, label="Base XGBoost")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title(
        f"Calibration Curve - Test Set (Base XGBoost)\nBrier: {brier_base:.4f} | Log loss: {ll_base:.4f}",
        fontsize=12,
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    base_calibration_path = FIGURE_DIR / "calibration_curve_test_base.png"
    plt.savefig(base_calibration_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {base_calibration_path}")

# 샘플별 예측 결과와 정오답 여부를 CSV로 저장합니다.
results_df = df_test_raw.copy()
results_df["true_label"] = y_test.to_numpy()
results_df["predicted_label"] = y_pred.astype(int)
results_df["predicted_label_name"] = [label_name_map[int(v)] for v in y_pred]
results_df["pred_prob_hemodialysis"] = y_pred_proba_all[:, 0]
results_df["pred_prob_peritoneal_dialysis"] = y_pred_proba
results_df["is_correct"] = (results_df["true_label"] == results_df["predicted_label"])
results_csv_path = REPORT_DIR / "test_predictions_with_correctness.csv"
results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
print(f"Saved: {results_csv_path}")


run_all_test_sample_shap_analysis(
    model=model,
    base_model=base_model,
    X_full=X_test,
    output_dir=str(SHAP_OUTPUT_DIR),
)

# 8. Test subgroup 성능 분석: 나이/성별/지역 메타데이터를 분석용으로만 사용합니다.

def subgroup_performance_report(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_1: np.ndarray,
    *,
    group_cols=("age>=65", "gender_male", "region_capital"),
    positive_label=1,
    min_group_size=10,
) -> pd.DataFrame:
    """Binary classification 결과를 subgroup별로 나누어 성능 지표를 계산합니다.

    - df_meta: subgroup 컬럼을 담은 DataFrame입니다.
    - y_true/y_pred: 실제 label과 예측 label입니다.
    - y_proba_1: class=1, 즉 복막투석 확률입니다.
    """
    rows = []
    n_total = len(y_true)

    for gcol in group_cols:
        if gcol not in df_meta.columns:
            rows.append({
                "group_col": gcol,
                "group_value": "MISSING_COLUMN",
                "n": 0,
                "ratio": np.nan,
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "auc": np.nan,
                "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
                "note": f"{gcol} not found in df_meta",
            })
            continue

        gvals = df_meta[gcol].copy()

        # subgroup 값이 비어 있으면 분석용 "missing" 그룹으로 유지합니다.
        gvals = gvals.where(~pd.isna(gvals), other="missing")

        for gv in sorted(gvals.unique(), key=lambda x: (str(type(x)), str(x))):
            idx = (gvals == gv).to_numpy()
            n = int(idx.sum())
            if n == 0:
                continue

            yt = y_true[idx]
            yp = y_pred[idx]
            yp1 = y_proba_1[idx]

            acc = accuracy_score(yt, yp)
            f1m = f1_score(yt, yp, average="macro")

            # AUC는 해당 subgroup 안에 두 class가 모두 있을 때만 계산할 수 있습니다.
            auc = np.nan
            if len(np.unique(yt)) == 2:
                auc = roc_auc_score(yt, yp1)

            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

            note = ""
            if n < min_group_size:
                note = f"small_n(<{min_group_size})"

            rows.append({
                "group_col": gcol,
                "group_value": gv,
                "n": n,
                "ratio": n / n_total,
                "accuracy": acc,
                "macro_f1": f1m,
                "auc": auc,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "note": note,
            })

    out = pd.DataFrame(rows)

    # 결과를 subgroup 컬럼과 값 기준으로 정렬해 CSV에서 바로 읽기 쉽게 만듭니다.
    if not out.empty:
        out["group_col"] = out["group_col"].astype(str)
        out["group_value"] = out["group_value"].astype(str)
        out = out.sort_values(["group_col", "group_value"]).reset_index(drop=True)

    return out


def print_subgroup_report(df_report: pd.DataFrame, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if df_report.empty:
        print("No subgroup results.")
        return
    show_cols = ["group_col", "group_value", "n", "ratio", "accuracy", "macro_f1", "auc", "tn", "fp", "fn", "tp", "note"]
    print(df_report[show_cols].to_string(index=False))

META_COLS = ["age>=65", "gender_male", "region_capital", "region_missing", "region_norm"]
df_meta_test = df_test[[c for c in META_COLS if c in df_test.columns]].copy()

subgroup_report_test = subgroup_performance_report(
    df_meta=df_meta_test,
    y_true=y_test.to_numpy(),
    y_pred=y_pred,
    y_proba_1=y_pred_proba,
    group_cols=("age>=65", "gender_male", "region_capital"),
    min_group_size=10,
)

print_subgroup_report(subgroup_report_test, title="Subgroup Performance (Test)")

subgroup_csv_path = REPORT_DIR / "subgroup_performance_test.csv"
subgroup_report_test.to_csv(subgroup_csv_path, index=False, encoding="utf-8")
print("Saved:", subgroup_csv_path)

print("\nComplete!")
