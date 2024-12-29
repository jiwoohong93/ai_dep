import pandas as pd
import numpy as np
import pickle
import os
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef
)
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

    deo = tpr_neg - tpr_pos
    return deo, tpr_neg, tpr_pos

def compute_and_print_metrics(y_true, y_pred, label_list=None):
    acc = accuracy_score(y_true, y_pred)
    
    metrics_dict = {}
    metrics_dict['accuracy'] = acc
    
    metrics_dict['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    
    metrics_dict['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    
    metrics_dict['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    
    metrics_dict['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics_dict['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics_dict['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    try:
        y_true_binarized = pd.get_dummies(y_true, columns=label_list)
        y_pred_binarized = pd.get_dummies(y_pred, columns=label_list)
        common_cols = y_true_binarized.columns.intersection(y_pred_binarized.columns)
        metrics_dict['roc_auc_macro'] = roc_auc_score(
            y_true_binarized[common_cols], y_pred_binarized[common_cols], average='macro'
        )
        
        metrics_dict['avg_precision_macro'] = average_precision_score(
            y_true_binarized[common_cols], y_pred_binarized[common_cols], average='macro'
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
    
    values = []

    print(f"\n=== ‘정책 반영 성능 평균 하락률’ (Difference from {base_key}) ===")
    for k, v in metrics_dict.items():
        if k == base_key or v is None:
            continue
        diff = base_value - v
        print(f"{k:25s} : {v:.4f} | Drop from {base_key} = {diff:.4f}")

        values.append(diff)
    print(f'Average difference: {sum(values) / len(values)}')
    values_pos = [i for i in values if i > 0]
    print(f'Average difference (only drop): {sum(values_pos) / len(values_pos)}')
    

def main():
    logging.info("=== Step 1: Setup & Load Data ===")
    
    SAMPLING_FRACTION = 0.02696435312
    data_path = './data.csv'
    
    if not os.path.exists(data_path):
        msg = f"Could not find {data_path}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    logging.info("Reading CSV with low_memory=False...")
    df = pd.read_csv(data_path, encoding='utf-8', delimiter=',', low_memory=False)

    # Strip whitespace
    df.columns = df.columns.str.strip()

    logging.info(f"DataFrame shape BEFORE sampling: {df.shape}")
    df = df.sample(frac=SAMPLING_FRACTION, random_state=42).reset_index(drop=True)
    logging.info(f"DataFrame shape AFTER sampling {SAMPLING_FRACTION*100:.1f}%: {df.shape}")

    logging.info("=== Step 2: Choose Target Column ===")
    
    target_col = '최종학력'
    if target_col not in df.columns:
        msg = (f"Target column '{target_col}' not found in DataFrame columns. "
               f"Columns found: {list(df.columns)}")
        logging.error(msg)
        raise KeyError(msg)
    
    initial_count = df.shape[0]
    df = df.dropna(subset=[target_col])
    after_dropping_count = df.shape[0]
    logging.info(
        f"Dropped {initial_count - after_dropping_count} rows with NaN in '{target_col}'. "
        f"Remaining rows: {after_dropping_count}"
    )

    df[target_col] = df[target_col].astype(str)

    threshold = 5
    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= threshold].index
    logging.info(
        f"Classes and their counts BEFORE filtering: {counts.to_dict()}\n"
        f"Keeping classes with at least {threshold} samples: {list(valid_classes)}"
    )
    df = df[df[target_col].isin(valid_classes)].reset_index(drop=True)
    if df.shape[0] < 2:
        msg = "Not enough data after filtering by class frequency."
        logging.error(msg)
        raise ValueError(msg)
    logging.info(f"DataFrame shape AFTER filtering classes: {df.shape}")

    logging.info("=== Step 3: Handle Class Imbalance ===")
    
    X_temp = df.drop(columns=[target_col])
    y_temp = df[target_col].copy()

    # Oversample
    from imblearn.over_sampling import SMOTE

    logging.info("Applying RandomOverSampler...")
    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X_temp, y_temp)

    

    logging.info(
        f"Class distribution AFTER oversampling: {pd.Series(y_balanced).value_counts().to_dict()}"
    )

    logging.info("=== Step 4: Choose Features ===")
    
    selected_features = [
        "전체경력",
        "희망연봉",
        "근무형태",
        "희망(경력)직종1",
        "희망(경력)업종1",
        "근무지역",
        "희망(경력)세부직종1",
        "희망(경력)직종2",
        "직무태그",
        "희망(경력)업종1",
        "최종연봉"
    ]
    selected_features = list(set(selected_features))

    logging.info(f"Selected features (candidate): {selected_features}")
    final_features = [col for col in selected_features if col in X_balanced.columns]
    logging.info(f"Final features actually in dataset: {final_features}")

    if not final_features:
        msg = ("No valid features found in the dataset for training. "
               "Please adjust `selected_features`.")
        logging.error(msg)
        raise ValueError(msg)

    X_balanced_final = X_balanced[final_features].copy()
    y_balanced_final = y_balanced.copy()

    protected_col = "학교명"
    if protected_col not in X_balanced.columns:
        logging.warning(f"'{protected_col}' not in columns. Creating a dummy protected_col.")
        X_balanced[protected_col] = "NoSchool"
    
    protected_series = X_balanced[protected_col].astype(str).apply(
        lambda x: 1 if "여자" in x else -1
    )

    logging.info("=== Step 6: Clean & Encode Features ===")

    X_balanced_final = X_balanced_final.fillna("missing")
    for c in X_balanced_final.columns:
        X_balanced_final[c] = X_balanced_final[c].astype(str)

    logging.info(f"Performing one-hot encoding on selected features...")
    X_encoded = pd.get_dummies(X_balanced_final, columns=X_balanced_final.columns)
    logging.info(f"Shape of X after get_dummies: {X_encoded.shape}")

    logging.info("=== Step 7: Split into Train/Test ===")

    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X_encoded, 
        y_balanced_final, 
        protected_series, 
        test_size=0.1, 
        random_state=42, 
        stratify=y_balanced_final
    )

    logging.info(
        f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape} | "
        f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}"
    )

    test_index_path = "test_indices.csv"
    pd.DataFrame({"test_index": X_test.index}).to_csv(test_index_path, index=False)
    logging.info(f"Test indices saved to '{test_index_path}'")

    logging.info("=== Step 8: Train a Model (RandomForest) ===")

    clf = RandomForestClassifier(n_estimators=140, max_depth=15, random_state=42)
    logging.info("Fitting the RandomForestClassifier on the training set...")
    clf.fit(X_train, y_train)

    logging.info("=== Step 9: Evaluate the Model ===")

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {acc:.4f}")
    print(f"\n=== Test Accuracy = {acc:.4f} ===")

    label_list = list(y_balanced_final.unique())  # possible classes
    metrics_dict = compute_and_print_metrics(y_test, y_pred, label_list=label_list)
    
    print("\n=== All Computed Metrics ===")
    for k, v in metrics_dict.items():
        if v is None:
            print(f"{k:25s} : None")
        else:
            print(f"{k:25s} : {v:.4f}")
    
    print_fairness_degradation(metrics_dict, base_key='accuracy')

    positive_label = "고등학교 졸업"
    deo_value, tpr_neg, tpr_pos = measure_deo(
        y_true=y_test,
        y_pred=y_pred,
        protected_group=prot_test,
        positive_label=positive_label
    )
    print(f"\n\n=== DEO (Difference of Equality of Opportunity) ===")
    print(f"Positive label = '{positive_label}'")
    print(f"TPR (unprotected group, a=-1) = {tpr_neg:.4f}")
    print(f"TPR (protected group, a=+1)   = {tpr_pos:.4f}")
    print(f"DEO = TPR(-1) - TPR(+1)       = {deo_value:.4f}")

    logging.info("=== Step 10: Save the Model & Encoding Info ===")

    model_path = "trained_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "dummy_columns": X_encoded.columns,
            "selected_features": final_features
        }, f)

    logging.info(f"Model saved to '{model_path}'")

if __name__ == "__main__":
    main()
