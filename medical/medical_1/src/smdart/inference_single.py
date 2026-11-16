"""저장된 calibrated XGBoost 모델로 단일 환자 예측을 수행합니다."""

import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from causal_intervention import CausalInterventionClassifier  # noqa: F401 (저장된 모델 로드에 필요)


# 경로 설정: 환경변수가 없으면 프로젝트 내부 기본 경로를 사용합니다.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
SINGLE_OUTPUT_DIR = OUTPUT_DIR / "inference" / "single"
REPORT_DIR = SINGLE_OUTPUT_DIR / "reports"
SHAP_OUTPUT_DIR = SINGLE_OUTPUT_DIR / "shap"

TEST_CSV_NAME = "df_merge_test_diagnosis_18_eng.csv"
MODEL_FILE = "calibrated_xgboost_model.pkl"
BASE_MODEL_FILE = "base_xgb_model.pkl"
FEATURE_FILE = "feature_names.pkl"

# 입력 설정: JSON 문자열/파일, CSV 한 행, 기본 test CSV 중 하나를 사용할 수 있습니다.
SINGLE_INPUT_JSON = os.getenv("SINGLE_INPUT_JSON")
SINGLE_INPUT_CSV = os.getenv("SINGLE_INPUT_CSV")
SINGLE_INPUT_INDEX = int(os.getenv("SINGLE_INPUT_INDEX", "0"))
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "1").lower() not in {"0", "false", "no"}
SHAP_TOP_N = int(os.getenv("SHAP_TOP_N", "15"))

for output_dir in (REPORT_DIR, SHAP_OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)


# 모델 label 정의입니다.
label_name_map = {
    0: "Hemodialysis (0)",
    1: "Peritoneal dialysis (1)",
}


def sanitize_feature_names(cols: pd.Index) -> pd.Index:
    """XGBoost가 허용하지 않는 feature name 문자를 train.py와 동일하게 치환합니다."""
    cols = cols.astype(str)
    cols = cols.str.replace("[", "(", regex=False)
    cols = cols.str.replace("]", ")", regex=False)
    cols = cols.str.replace("<", "less_than", regex=False)
    return cols


def fill_missing_like_training(X: pd.DataFrame) -> pd.DataFrame:
    """학습 스크립트와 같은 방식으로 단일 입력의 결측값을 채웁니다."""
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


def load_patient_row():
    """JSON, CSV 또는 기본 test CSV에서 단일 환자 입력 한 건을 불러옵니다."""
    if SINGLE_INPUT_JSON:
        json_path = Path(SINGLE_INPUT_JSON)
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            source = {"type": "json_file", "path": str(json_path)}
        else:
            payload = json.loads(SINGLE_INPUT_JSON)
            source = {"type": "json_string"}
        return payload, source

    csv_path = Path(SINGLE_INPUT_CSV) if SINGLE_INPUT_CSV else DATA_DIR / TEST_CSV_NAME
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")
    if SINGLE_INPUT_INDEX < 0 or SINGLE_INPUT_INDEX >= len(df):
        raise IndexError(f"SINGLE_INPUT_INDEX={SINGLE_INPUT_INDEX} is out of range for {csv_path}")

    row = df.iloc[SINGLE_INPUT_INDEX].to_dict()
    source = {
        "type": "csv_row",
        "path": str(csv_path),
        "row_index": SINGLE_INPUT_INDEX,
    }
    return row, source


def align_patient_features(patient_data: dict, feature_names) -> pd.DataFrame:
    """단일 환자 입력을 학습 당시 feature 순서와 동일하게 정렬합니다."""
    patient_df = pd.DataFrame([patient_data])
    patient_df.columns = sanitize_feature_names(patient_df.columns)

    for c in feature_names:
        if c not in patient_df.columns:
            patient_df[c] = np.nan

    patient_df = patient_df[feature_names].copy()
    return fill_missing_like_training(patient_df)


def save_single_sample_shap(model, base_model, patient_df: pd.DataFrame) -> None:
    """단일 환자 입력에 대한 SHAP waterfall plot과 feature 기여도 CSV를 저장합니다."""
    if base_model is None:
        print("Base model not found. SHAP 분석을 건너뜁니다.")
        return
    if not ENABLE_SHAP:
        print("ENABLE_SHAP이 비활성화되어 SHAP 분석을 건너뜁니다.")
        return

    # 보호 속성에 개입한 예측에 대한 설문 문항별 SHAP 값
    shap_exp = model.explain(base_model, patient_df)
    shap_values = shap_exp.values[0]

    shap_df = pd.DataFrame({
        "feature": patient_df.columns,
        "feature_value": patient_df.iloc[0].values,
        "shap_value": shap_values,
        "abs_shap_value": np.abs(shap_values),
    }).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

    shap_csv_path = SHAP_OUTPUT_DIR / "shap_single_patient.csv"
    shap_df.to_csv(shap_csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_exp[0], max_display=SHAP_TOP_N, show=False)
    plt.tight_layout()
    shap_plot_path = SHAP_OUTPUT_DIR / "shap_single_patient.png"
    plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP CSV 저장: {shap_csv_path}")
    print(f"SHAP plot 저장: {shap_plot_path}")


def main() -> None:
    """모델을 로드하고 단일 환자 예측 결과를 저장합니다."""
    model_path = MODEL_DIR / MODEL_FILE
    base_model_path = MODEL_DIR / BASE_MODEL_FILE
    feature_path = MODEL_DIR / FEATURE_FILE

    model = joblib.load(model_path)
    base_model = joblib.load(base_model_path) if base_model_path.exists() else None
    feature_names = joblib.load(feature_path)

    patient_data, source = load_patient_row()
    patient_df = align_patient_features(patient_data, feature_names)

    start_time = time.time()
    pred_class = int(model.predict(patient_df)[0])
    proba = model.predict_proba(patient_df)[0]
    inference_time_ms = (time.time() - start_time) * 1000.0

    result = {
        "input_source": source,
        "predicted_class": pred_class,
        "predicted_label": label_name_map.get(pred_class, str(pred_class)),
        "probabilities_raw": {
            "Hemodialysis": float(proba[0]),
            "Peritoneal dialysis": float(proba[1]),
        },
        "probabilities_pct": {
            "Hemodialysis": f"{proba[0] * 100:.2f}%",
            "Peritoneal dialysis": f"{proba[1] * 100:.2f}%",
        },
        "inference_time_ms": inference_time_ms,
    }

    if "labels" in patient_data:
        true_label = int(patient_data["labels"])
        result["true_label"] = true_label
        result["true_label_name"] = label_name_map.get(true_label, str(true_label))
        result["is_correct"] = bool(true_label == pred_class)

    result_path = REPORT_DIR / "single_prediction_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("Single Patient Prediction")
    print("=" * 80)
    print(f"입력 source: {source}")
    if "true_label" in result:
        print(f"실제 label: {result['true_label']} - {result['true_label_name']}")
    print(f"예측 class: {result['predicted_class']} - {result['predicted_label']}")
    print("예측 확률:")
    for name, pct in result["probabilities_pct"].items():
        print(f"  {name}: {pct}")
    print(f"추론 시간: {inference_time_ms:.2f} ms")
    print(f"결과 저장: {result_path}")

    save_single_sample_shap(model, base_model, patient_df)


if __name__ == "__main__":
    main()
