"""전처리된 학습 데이터로 calibrated XGBoost 모델을 학습하고 결과물을 저장합니다.

모델은 설문 응답과 보호 속성(나이, 성별, 지역)을 함께 입력받아 학습하며,
예측에는 보호 속성에 대한 인과 개입(causal_intervention.py)을 적용합니다.
"""

import os
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, confusion_matrix, f1_score, log_loss, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from causal_intervention import CausalInterventionClassifier, PROTECTED_FEATURES, protected_distribution

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "processed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
TRAIN_OUTPUT_DIR = OUTPUT_DIR / "train"
FIGURE_DIR = TRAIN_OUTPUT_DIR / "figures"
REPORT_DIR = TRAIN_OUTPUT_DIR / "reports"

INPUT_FILE_PATH = DATA_DIR / "df_merge_train_diagnosis_18_eng.csv"

for output_dir in (MODEL_DIR, FIGURE_DIR, REPORT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

# 1. XGBoost CUDA 사용 가능 여부 확인
def check_gpu_availability():
    """작은 테스트 모델을 학습해 현재 환경에서 XGBoost GPU 학습이 가능한지 확인합니다."""
    try:
        test_model = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
        test_X = np.random.rand(10, 5)
        test_y = np.random.randint(0, 2, 10)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            test_model.fit(test_X, test_y)
        # GPU가 없으면 XGBoost는 예외 대신 경고를 내고 CPU로 fallback합니다.
        messages = " ".join(str(r.message) for r in records)
        if "No visible GPU" in messages or "not compiled with CUDA" in messages or "device to CPU" in messages:
            return False
        return True
    except Exception as e:
        print(f"GPU not available: {e}")
        return False


GPU_AVAILABLE = check_gpu_availability()
print("=" * 80)
print(f"GPU Available: {GPU_AVAILABLE}")
print("=" * 80)

# 2. 그림 저장을 위한 Matplotlib 기본 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 3. 학습 데이터 로드
print("\n" + "=" * 80)
print("Loading Data...")
print("=" * 80)
df = pd.read_csv(INPUT_FILE_PATH)

print(f"\nData shape: {df.shape}")
print(f"\nLabel distribution (raw counts):")
print(df['labels'].value_counts().sort_index())
print(f"\nLabel ratio:")
print(df['labels'].value_counts(normalize=True).sort_index())

# 4. 정답 라벨과 학습 feature 분리
EXCLUDE_COLS = [
    "ParticipantID",
    "labels",
    "age>=65",
    "gender_male",
    "region_capital",
    "region_missing",
    "region_norm",
]

y = pd.to_numeric(df["labels"], errors="coerce").astype(int)

def should_exclude_feature(col: str) -> bool:
    c = col.strip()
    if c in EXCLUDE_COLS:
        return True
    lc = c.lower()

    # 인구통계/지역 메타데이터는 컬럼명이 조금 달라도 설문 feature에서 제외합니다.
    if lc.startswith('region_'):
        return True
    if 'region' in lc and ('capital' in lc or 'missing' in lc or 'bucket' in lc or 'norm' in lc):
        return True
    if 'age' in lc and ('>=65' in lc or 'ge_65' in lc or '65' in lc or 'bin' in lc):
        return True

    return False

# X: 설문 feature (추론 시 입력), A: 보호 속성 (학습 시에만 사용, 추론 시에는 개입)
feature_cols = [c for c in df.columns if not should_exclude_feature(c)]
X = df[feature_cols].copy()
protected_cols = [c for c in PROTECTED_FEATURES if c in df.columns]
A = df[protected_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

# XGBoost가 허용하지 않는 feature name 문자를 저장 전 한 번 정리합니다.
X.columns = X.columns.str.replace('[', '(', regex=False)
X.columns = X.columns.str.replace(']', ')', regex=False)
X.columns = X.columns.str.replace('<', 'less_than', regex=False)

print(f"\nNumber of survey features: {X.shape[1]}")
print(f"Protected attributes (intervened at inference): {protected_cols}")
print(f"Number of samples: {X.shape[0]}")
print("\nExcluded columns:")
print([c for c in EXCLUDE_COLS if c in df.columns])

# 5. 결측값 처리
print(f"\nChecking missing values in X:")
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print(missing_counts[missing_counts > 0].head(50))
    X = X.apply(pd.to_numeric, errors='ignore')
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    for c in X.columns:
        if X[c].isnull().any():
            mode = X[c].mode(dropna=True)
            X[c] = X[c].fillna(mode.iloc[0] if len(mode) > 0 else 0)
    print("Missing values filled (numeric: median, others: mode).")
else:
    print("No missing values")

# 6. 이진 분류 라벨 정의
label_name_map = {
    0: 'Hemodialysis (0)',
    1: 'Peritoneal dialysis (1)',
}
unique_labels = np.sort(np.unique(y))
target_names = [label_name_map[int(l)] for l in unique_labels]

# 7. 이진 분류 XGBoost 모델 설정
print("\n" + "=" * 80)
print("Starting XGBoost Model Training (Binary)")
if GPU_AVAILABLE:
    print("Using GPU acceleration (gpu_hist)")
else:
    print("Using CPU (hist)")
print("=" * 80)

xgb_params = {
    "objective": "binary:logistic",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda" if GPU_AVAILABLE else "cpu"
}

base_model = xgb.XGBClassifier(**xgb_params)

# 학습 입력: 설문 feature + 보호 속성
X_full = pd.concat([X, A], axis=1)
intervention_combinations, intervention_weights = protected_distribution(A)

# 8. 기본 XGBoost 모델 교차검증
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nBase XGBoost Model Cross Validation Evaluation:")
start_time = time.time()
cv_scores = cross_val_score(base_model, X_full, y, cv=cv, scoring='accuracy')
cv_time = time.time() - start_time

print(f"CV Accuracy scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"CV Training time: {cv_time:.2f} seconds")

# 9. CalibratedClassifierCV로 확률 보정 모델 학습
print("\n" + "=" * 80)
print("Training Probability Calibration Model with CalibratedClassifierCV")
print("=" * 80)

calibrated_estimator = CalibratedClassifierCV(
    base_model,
    method='sigmoid',
    cv=5,
    ensemble=True
)

print("\nTraining calibrated model with full dataset...")
start_time = time.time()
calibrated_estimator.fit(X_full, y)
training_time = time.time() - start_time

# 예측 시 보호 속성에 개입하여 학습 데이터의 분포로 가중 평균하는 최종 모델
calibrated_model = CausalInterventionClassifier(
    calibrated_estimator,
    survey_features=X.columns.tolist(),
    protected_features=protected_cols,
    combinations=intervention_combinations,
    weights=intervention_weights,
)
print(f"Training complete! Time: {training_time:.2f} seconds")

# 10. 학습 데이터 기준 모델 성능 평가
y_pred = calibrated_model.predict(X)
y_pred_proba = calibrated_model.predict_proba(X)[:, 1]

print("\n" + "=" * 80)
print("Model Performance Evaluation (Train)")
print("=" * 80)

acc = accuracy_score(y, y_pred)
f1_macro = f1_score(y, y_pred, average='macro')
auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y)) == 2 else np.nan
brier = brier_score_loss(y, y_pred_proba)
ll = log_loss(y, y_pred_proba)

print(f"\nAccuracy: {acc:.4f}")
print(f"Macro F1: {f1_macro:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print(f"Brier score: {brier:.4f}")
print(f"Log loss:    {ll:.4f}")

print("\nClassification Report:")
print(classification_report(y, y_pred, labels=unique_labels, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y, y_pred, labels=unique_labels)
print(cm)

# 학습 데이터 confusion matrix 저장
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names
)
plt.title(
    'Confusion Matrix - Calibrated XGBoost'
    + (' (GPU)' if GPU_AVAILABLE else ' (CPU)')
    + f'\nAccuracy: {acc:.4f} ({acc*100:.2f}%)'
    + f' | Macro F1: {f1_macro:.4f}'
    + f' | AUC: {auc:.4f}',
    fontsize=13
)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

confusion_matrix_file_path = FIGURE_DIR / 'confusion_matrix_train.png'
plt.savefig(confusion_matrix_file_path, dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved:", confusion_matrix_file_path)

# 학습 데이터 calibration curve 저장
prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10, strategy='quantile')

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibrated XGBoost')
plt.xlabel('Mean predicted probability')
plt.ylabel('Observed positive rate')
plt.title(
    'Calibration Curve - Train'
    + (' (GPU)' if GPU_AVAILABLE else ' (CPU)')
    + f'\nBrier: {brier:.4f} | Log loss: {ll:.4f}',
    fontsize=12
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

calibration_curve_file_path = FIGURE_DIR / 'calibration_curve_train.png'
plt.savefig(calibration_curve_file_path, dpi=300, bbox_inches='tight')
plt.close()
print("Calibration curve saved:", calibration_curve_file_path)

# 학습 데이터 subgroup 성능 분석: 학습 feature에서 제외한 나이/성별/지역 메타데이터를 분석용으로만 사용합니다.

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
    """
    Binary classification 결과를 subgroup별로 나누어 성능 지표를 계산합니다.

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
df_meta = df[[c for c in META_COLS if c in df.columns]].copy()

subgroup_report_train = subgroup_performance_report(
    df_meta=df_meta,
    y_true=y.to_numpy(),
    y_pred=y_pred,
    y_proba_1=y_pred_proba,
    group_cols=("age>=65", "gender_male", "region_capital"),
    min_group_size=10,
)

print_subgroup_report(subgroup_report_train, title="Subgroup Performance (Train)")

# 후속 검토를 위해 subgroup 결과를 CSV로 저장합니다.
subgroup_csv_path = REPORT_DIR / "subgroup_performance_train.csv"
subgroup_report_train.to_csv(subgroup_csv_path, index=False, encoding="utf-8")
print("Saved:", subgroup_csv_path)
   

# 11. Feature importance 분석
print("\n" + "=" * 80)
print("Feature Importance Analysis")
print("=" * 80)

base_model.fit(X_full, y)

# SHAP 분석에는 calibration 전 base XGBoost 모델이 더 적합하므로 별도로 저장
base_model_path = MODEL_DIR / "base_xgb_model.pkl"
joblib.dump(base_model, base_model_path)
print("Base XGBoost model saved:", base_model_path)

feature_importance = pd.DataFrame({
    'feature': X_full.columns,
    'importance': base_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))

plt.figure(figsize=(12, 14))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'], fontsize=9)
plt.xlabel('Importance')
plt.title(
    f'All Feature Importances - XGBoost ({len(feature_importance)} features)'
    + (' (GPU)' if GPU_AVAILABLE else ' (CPU)')
)
plt.gca().invert_yaxis()
plt.tight_layout()
feature_importance_file_path = FIGURE_DIR / 'feature_importance.png'
plt.savefig(feature_importance_file_path, dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved:", feature_importance_file_path)

# 12. 간단한 예측 helper
def predict_treatment(model, input_data, feature_names):
    """
    단일 입력에 대해 투석 방식 class와 class별 확률을 반환합니다.

    Class 정의:
        0: Hemodialysis
        1: Peritoneal dialysis
    """
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = pd.DataFrame([input_data], columns=feature_names)

    pred_class = int(model.predict(input_df)[0])
    pred_proba = model.predict_proba(input_df)[0]  # [혈액투석 확률, 복막투석 확률]

    class_names = {
        0: 'Hemodialysis',
        1: 'Peritoneal dialysis',
    }

    prob_dict = {
        class_names[0]: f"{pred_proba[0]*100:.2f}%",
        class_names[1]: f"{pred_proba[1]*100:.2f}%",
    }

    result = {
        'predicted_class': pred_class,
        'predicted_label': class_names.get(pred_class, f'Class {pred_class}'),
        'probabilities': prob_dict,
        'raw_probabilities': {0: float(pred_proba[0]), 1: float(pred_proba[1])}
    }
    return result

# 13. 학습 데이터 일부로 예측 예시 출력
print("\n" + "=" * 80)
print("Sample Prediction Examples")
print("=" * 80)

for i in range(min(5, len(X))):
    sample = X.iloc[i]
    true_label = int(y.iloc[i])
    result = predict_treatment(calibrated_model, sample, X.columns)

    print(f"\nSample #{i+1} (True label: {true_label} - {label_name_map[true_label]})")
    print(f"  Predicted class: {result['predicted_class']} - {result['predicted_label']}")
    print("  Probability distribution:")
    for pref, prob in result['probabilities'].items():
        print(f"    {pref}: {prob}")

# 14. 평균 feature 값으로 만든 가상 환자 예측 예시
print("\n" + "=" * 80)
print("New Data Inference Example")
print("=" * 80)

new_patient = X.mean(numeric_only=True).to_dict()
# 누락된 feature가 있으면 0으로 채워 모델 입력 순서를 맞춥니다.
for c in X.columns:
    if c not in new_patient:
        new_patient[c] = 0

result = predict_treatment(calibrated_model, new_patient, X.columns)

print("\nVirtual new patient (average feature values)")
print(f"Predicted class: {result['predicted_class']} - {result['predicted_label']}")
print("Probability distribution:")
for pref, prob in result['probabilities'].items():
    print(f"  {pref}: {prob}")

# 15. 모델과 학습 메타데이터 저장
print("\n" + "=" * 80)
print("Saving Models")
print("=" * 80)

model_filename = 'calibrated_xgboost_model.pkl'
model_file_path = MODEL_DIR / model_filename
joblib.dump(calibrated_model, model_file_path)
print(f"Calibrated model saved: {model_file_path}")

feature_names_file_path = MODEL_DIR / 'feature_names.pkl'
joblib.dump(X.columns.tolist(), feature_names_file_path)
print("Feature names saved: feature_names.pkl")

training_info = {
    'gpu_used': GPU_AVAILABLE,
    'cv_time': cv_time,
    'training_time': training_time,
    'accuracy_train': acc,
    'f1_macro_train': f1_macro,
    'auc_train': auc,
    'brier_score_train': float(brier),
    'log_loss_train': float(ll),
    'cv_mean_accuracy': float(cv_scores.mean()),
    'cv_std_accuracy': float(cv_scores.std()),
    'xgb_params': xgb_params,
    'label_name_map': label_name_map,
    'survey_features': X.columns.tolist(),
    'protected_features': protected_cols,
    'intervention_combinations': intervention_combinations.tolist(),
    'intervention_weights': intervention_weights.tolist(),
}
training_info_file_path = MODEL_DIR / 'training_info.pkl'
joblib.dump(training_info, training_info_file_path)
print("Training info saved: training_info.pkl")

print("\n" + "=" * 80)
print("Performance Summary")
print("=" * 80)
print(f"GPU Used: {GPU_AVAILABLE}")
print(f"CV Time: {cv_time:.2f} seconds")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Total Time: {cv_time + training_time:.2f} seconds")
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Train Accuracy: {acc:.4f}")
print(f"Train Macro F1: {f1_macro:.4f}")
print(f"Train AUC: {auc:.4f}")
print(f"Train Brier score: {brier:.4f}")
print(f"Train Log loss: {ll:.4f}")

print("\n" + "=" * 80)
print("Complete!")
print("=" * 80)
