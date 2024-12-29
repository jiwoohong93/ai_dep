import pandas as pd
import numpy as np
import pickle
import os
import logging

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    balanced_accuracy_score, 
    cohen_kappa_score, 
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from imblearn.over_sampling import RandomOverSampler

def measure_deo(y_true, y_pred, protected_group, positive_label):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    protected_group = np.array(protected_group)

    mask_neg = (protected_group == -1)
    mask_pos = (protected_group == 1)

    actual_positives_neg = (y_true[mask_neg] == positive_label)
    denom_neg = np.sum(actual_positives_neg)
    if denom_neg == 0:
        tpr_neg = 0.0
    else:
        tpr_neg = np.sum(
            (y_true[mask_neg] == positive_label) & (y_pred[mask_neg] == positive_label)
        ) / denom_neg

    actual_positives_pos = (y_true[mask_pos] == positive_label)
    denom_pos = np.sum(actual_positives_pos)
    if denom_pos == 0:
        tpr_pos = 0.0
    else:
        tpr_pos = np.sum(
            (y_true[mask_pos] == positive_label) & (y_pred[mask_pos] == positive_label)
        ) / denom_pos

    deo = abs(tpr_neg - tpr_pos)
    return deo, tpr_neg, tpr_pos


def compute_and_print_metrics(y_true, y_pred, label_list=None):
    metrics_dict = {}
    acc = accuracy_score(y_true, y_pred)
    metrics_dict['accuracy'] = acc
    
    metrics_dict['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics_dict['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics_dict['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics_dict['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics_dict['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics_dict['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

    try:
        # Binarize for multi-class AUC calculation
        y_true_binarized = pd.get_dummies(y_true, columns=label_list)
        y_pred_binarized = pd.get_dummies(y_pred, columns=label_list)
        common_cols = y_true_binarized.columns.intersection(y_pred_binarized.columns)
        metrics_dict['roc_auc_macro'] = roc_auc_score(
            y_true_binarized[common_cols],
            y_pred_binarized[common_cols],
            average='macro'
        )
        metrics_dict['avg_precision_macro'] = average_precision_score(
            y_true_binarized[common_cols],
            y_pred_binarized[common_cols],
            average='macro'
        )
    except Exception as e:
        logging.warning("Could not compute ROC AUC or Average Precision: {}".format(e))
        metrics_dict['roc_auc_macro'] = None
        metrics_dict['avg_precision_macro'] = None

    return metrics_dict

def print_fairness_degradation(metrics_dict, base_key='accuracy'):
    base_value = metrics_dict.get(base_key, None)
    if base_value is None:
        print(f"[WARN] '{base_key}' not found in metrics. Skipping fairness degradation.")
        return
    
    print(f"\n=== ‘정책 반영 성능 평균 하락률’ (Difference from {base_key}) ===")
    values = []
    for k, v in metrics_dict.items():
        if k == base_key or v is None:
            continue
        diff = base_value - v
        print(f"{k:25s} : {v:.4f} | Drop from {base_key} = {diff:.4f}")
        values.append(diff)
    if len(values) > 0:
        avg_diff = sum(values) / len(values)
        print(f'Average difference: {avg_diff:.4f}')
        only_drops = [d for d in values if d > 0]
        if len(only_drops) > 0:
            avg_only_drops = sum(only_drops) / len(only_drops)
            print(f'Average difference (only drop): {avg_only_drops:.4f}')


def evaluate_pipeline(
    model_path,
    data_csv,
    test_index_csv,
    sampling_fraction,
    target_col,
    threshold,
    protected_col,
    selected_features
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file: '{model_path}'")
    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)
    clf = saved_data["model"]
    dummy_columns = saved_data["dummy_columns"]
    features_used = saved_data["selected_features"]
    
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Could not find data CSV: '{data_csv}'")
    df = pd.read_csv(data_csv, encoding='utf-8', delimiter=',', low_memory=False)
    df.columns = df.columns.str.strip()

    df = df.sample(frac=sampling_fraction, random_state=42).reset_index(drop=True)

    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(str)

    # filter out classes with < threshold
    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= threshold].index
    df = df[df[target_col].isin(valid_classes)].reset_index(drop=True)
    if df.shape[0] < 2:
        raise ValueError("Not enough data after filtering by class frequency.")

    X_temp = df.drop(columns=[target_col])
    y_temp = df[target_col].copy()

    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X_temp, y_temp)

    # protected attribute
    if protected_col not in X_balanced.columns:
        X_balanced[protected_col] = "NoSchool"  # fallback
    prot_series = X_balanced[protected_col].astype(str).apply(
        lambda x: 1 if "여자" in x else -1
    )

    # keep only selected features
    X_balanced_final = X_balanced[features_used].copy()
    X_balanced_final = X_balanced_final.fillna("missing")
    for c in X_balanced_final.columns:
        X_balanced_final[c] = X_balanced_final[c].astype(str)
    X_encoded = pd.get_dummies(X_balanced_final, columns=X_balanced_final.columns)

    # load test indices
    if not os.path.exists(test_index_csv):
        raise FileNotFoundError(
            f"Could not find test index CSV: '{test_index_csv}'."
        )
    test_idx_df = pd.read_csv(test_index_csv)
    test_indices = test_idx_df["test_index"].tolist()

    X_test = X_encoded.loc[test_indices]
    y_test = y_balanced.loc[test_indices]
    prot_test = prot_series.loc[test_indices]

    for col in dummy_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[dummy_columns]

    # predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc, y_test, y_pred, prot_test, dummy_columns


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # filenames for the "fair" pipeline
    fair_model_path = "trained_model_fair.pkl"
    fair_data_csv = "data.csv"
    fair_test_index_csv = "test_indices_fair.csv"

    # filenames for the "unfair" pipeline
    unfair_model_path = "trained_model_unfair.pkl"
    unfair_data_csv = "data.csv"
    unfair_test_index_csv = "test_indices_unfair.csv"

    # shared parameters (adjust if your training code used different ones)
    SAMPLING_FRACTION = 0.02696435312
    TARGET_COL = '최종학력'
    THRESHOLD = 5
    PROTECTED_COL = '학교명'

    ########################################################################
    # 1. Evaluate the FAIR Model
    ########################################################################
    fair_acc, fair_y_test, fair_y_pred, fair_prot_test, fair_dummy_cols = evaluate_pipeline(
        model_path=fair_model_path,
        data_csv=fair_data_csv,
        test_index_csv=fair_test_index_csv,
        sampling_fraction=SAMPLING_FRACTION,
        target_col=TARGET_COL,
        threshold=THRESHOLD,
        protected_col=PROTECTED_COL,
        selected_features=[]
    )

    print(f"\n=== Fair Model Test Accuracy = {fair_acc:.4f} ===")
    label_list_fair = list(fair_y_test.unique())
    metrics_dict_fair = compute_and_print_metrics(fair_y_test, fair_y_pred, label_list=label_list_fair)

    positive_label = '고등학교 졸업'
    deo_val, tpr_neg, tpr_pos = measure_deo(fair_y_test, fair_y_pred, fair_prot_test, positive_label)
    print(f"\n\n=== DEO (Fair Model) ===")
    print(f"Positive label: '{positive_label}'")
    print(f"TPR (unprotected) = {tpr_neg:.4f}")
    print(f"TPR (protected)   = {tpr_pos:.4f}")
    print(f"DEO = {deo_val:.4f}")

    ########################################################################
    # 2. Evaluate the UNFAIR Model
    ########################################################################
    unfair_acc, unfair_y_test, unfair_y_pred, _, _ = evaluate_pipeline(
        model_path=unfair_model_path,
        data_csv=unfair_data_csv,
        test_index_csv=unfair_test_index_csv,
        sampling_fraction=SAMPLING_FRACTION,
        target_col=TARGET_COL,
        threshold=THRESHOLD,
        protected_col=PROTECTED_COL,
        selected_features=[]
    )
    print(f"\n=== Unfair Model Test Accuracy = {unfair_acc:.4f} ===")

    ########################################################################
    # 3. Print Accuracy Difference (Fair - Unfair)
    ########################################################################
    acc_diff = abs(fair_acc - unfair_acc)
    print("\n=== Accuracy Difference (Fair - Unfair) ===")
    print(f"Fair Model Accuracy   : {fair_acc:.4f}")
    print(f"Unfair Model Accuracy : {unfair_acc:.4f}")
    print(f"Difference            : {acc_diff:.4f}")

if __name__ == "__main__":
    main()
