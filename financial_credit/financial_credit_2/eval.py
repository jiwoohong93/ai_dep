import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib

def main():
    ###############################################################################
    # 1) Load Artifacts
    ###############################################################################
    best_gbclf_smote = joblib.load('best_gbclf_smote.joblib')
    best_gbclf_ros   = joblib.load('best_gbclf_ros.joblib')
    
    label_encoder          = joblib.load('label_encoder.joblib')
    categorical_encoders   = joblib.load('categorical_encoders.joblib')
    scaler_smote           = joblib.load('scaler_smote.joblib')
    scaler_ros             = joblib.load('scaler_ros.joblib')
    numerical_cols         = joblib.load('numerical_cols.joblib')
    categorical_cols       = joblib.load('categorical_cols.joblib')
    
    ###############################################################################
    # 2) Load & Preprocess Data
    ###############################################################################
    data = pd.read_csv('./data/german.csv')
    data.replace('NA', np.nan, inplace=True)

    X = data.drop('Risk', axis=1)
    y = data['Risk']

    # Apply the label encoder to y
    y_encoded = label_encoder.transform(y)  # since we saved the label encoder from train time

    # Apply the categorical encoders to the X
    for col in categorical_cols:
        X[col] = categorical_encoders[col].transform(X[col].astype(str))

    ###############################################################################
    # 3) Reconstruct With Fairness Test Set
    ###############################################################################
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y_encoded)

    # Must use the same train/test split as in train_and_save.py
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
        X_smote, y_smote, test_size=0.2, random_state=1
    )

    # Scale the numeric columns using the loaded scaler_smote
    X_test_smote[numerical_cols] = scaler_smote.transform(X_test_smote[numerical_cols])

    ###############################################################################
    # 4) Reconstruct Without Fairness Test Set
    ###############################################################################
    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X, y_encoded)

    X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(
        X_ros, y_ros, test_size=0.2, random_state=1
    )

    # Scale the numeric columns using the loaded scaler_ros
    X_test_ros[numerical_cols] = scaler_ros.transform(X_test_ros[numerical_cols])

    ###############################################################################
    # 5) Evaluate the Loaded Model
    ###############################################################################
    y_pred_smote = best_gbclf_smote.predict(X_test_smote)
    accuracy_smote = accuracy_score(y_test_smote, y_pred_smote)

    print("=== GradientBoostingClassifier (With SMOTE) ===")
    print("Accuracy:", accuracy_smote)
    print("Best Parameters (from model):", best_gbclf_smote.get_params())
    print("\nClassification Report:")
    print(classification_report(y_test_smote, y_pred_smote))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_smote, y_pred_smote))

    # Cross-validation accuracy (SMOTE)
    cv_scores_smote = cross_val_score(best_gbclf_smote, X_smote, y_smote, cv=5, scoring='accuracy')
    print("\nCross-validation Accuracy (With SMOTE): %.2f (+/- %.2f)" %
          (cv_scores_smote.mean(), cv_scores_smote.std()))

    ###############################################################################
    # 6) Evaluate the Loaded Base Model
    ###############################################################################
    y_pred_ros = best_gbclf_ros.predict(X_test_ros)
    accuracy_ros = accuracy_score(y_test_ros, y_pred_ros)

    print("\n=== GradientBoostingClassifier (Random Oversampling) ===")
    print("Accuracy:", accuracy_ros)
    print("Best Parameters (from model):", best_gbclf_ros.get_params())
    print("\nClassification Report:")
    print(classification_report(y_test_ros, y_pred_ros))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_ros, y_pred_ros))

    # Cross-validation accuracy (ROS)
    cv_scores_ros = cross_val_score(best_gbclf_ros, X_ros, y_ros, cv=5, scoring='accuracy')
    print("\nCross-validation Accuracy (Random Oversampling): %.2f (+/- %.2f)" %
          (cv_scores_ros.mean(), cv_scores_ros.std()))

    ###############################################################################
    # 7) 정책 반영 성능 평균 하락률
    ###############################################################################
    policy_drop_ratio = (accuracy_ros - accuracy_smote) / accuracy_ros * 100
    print(f"\n정책 반영 성능 평균 하락률 (Test Set): {policy_drop_ratio:.2f}%")

    mean_ros = cv_scores_ros.mean()
    mean_smote = cv_scores_smote.mean()
    policy_drop_ratio_cv = (mean_ros - mean_smote) / mean_ros * 100
    print(f"정책 반영 성능 평균 하락률 (CV): {policy_drop_ratio_cv:.2f}%")

    ###############################################################################
    # 8) Compute DEO 
    ###############################################################################
    # Protected attribute: 'Sex' => male = 1, female = -1
    sex_encoder = categorical_encoders['Sex']
    classes_sex = list(sex_encoder.classes_)
    male_index = classes_sex.index('male')
    female_index = classes_sex.index('female')

    test_predictions_smote = y_pred_smote

    a_smote = X_test_smote['Sex'].apply(lambda x: 1 if x == male_index else -1)

    good_mask = (y_test_smote == 1)
    pred_good_mask = (test_predictions_smote == 1)

    female_good_mask = (a_smote == -1) & good_mask
    male_good_mask = (a_smote == 1) & good_mask

    tpr_female = pred_good_mask[female_good_mask].mean()  # among female & actual=1
    tpr_male   = pred_good_mask[male_good_mask].mean()    # among male & actual=1
    DEO = abs(tpr_female - tpr_male)

    print("\n=== Fairness Metrics (With SMOTE) ===")
    print(f"TPR (Female) = {tpr_female:.4f}")
    print(f"TPR (Male)   = {tpr_male:.4f}")
    print(f"DEO          = {DEO:.4f}")

    print("\n=== End of test script ===")

if __name__ == "__main__":
    main()
