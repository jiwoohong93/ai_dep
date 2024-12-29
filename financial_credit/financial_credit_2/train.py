import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler  # (NEW) RandomOverSampler
import joblib

###############################################################################
# 1) Load & Preprocess Data
###############################################################################
# Load the dataset
data = pd.read_csv('./data/german.csv')

# Replace 'NA' strings with NaN
data.replace('NA', np.nan, inplace=True)

# Define features and target
X = data.drop('Risk', axis=1)
y = data['Risk']

# Encode the target variable ('good'->1, 'bad'->0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Encode categorical features
categorical_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    X[col] = categorical_encoders[col].fit_transform(X[col].astype(str))

###############################################################################
# 2) Model with Fairness
###############################################################################
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y_encoded)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
    X_smote, y_smote, test_size=0.2, random_state=1
)

scaler = StandardScaler()
X_train_smote[numerical_cols] = scaler.fit_transform(X_train_smote[numerical_cols])
X_test_smote[numerical_cols] = scaler.transform(X_test_smote[numerical_cols])

param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.05],
    'max_depth': [5],
    'subsample': [0.8],
    'min_samples_split': [2,],
    'min_samples_leaf': [1]
}

gbclf_smote = GradientBoostingClassifier(random_state=42)
grid_search_smote = GridSearchCV(estimator=gbclf_smote,
                                 param_grid=param_grid,
                                 scoring='accuracy',
                                 cv=5, n_jobs=-1)
grid_search_smote.fit(X_train_smote, y_train_smote)

best_gbclf_smote = grid_search_smote.best_estimator_
y_pred_smote = best_gbclf_smote.predict(X_test_smote)
accuracy_smote = accuracy_score(y_test_smote, y_pred_smote)

print("=== GradientBoostingClassifier (With SMOTE) ===")
print("Accuracy:", accuracy_smote)
print("Best Parameters:", grid_search_smote.best_params_)
print("\nClassification Report:")
print(classification_report(y_test_smote, y_pred_smote))
print("Confusion Matrix:")
print(confusion_matrix(y_test_smote, y_pred_smote))

cv_scores_smote = cross_val_score(best_gbclf_smote, X_smote, y_smote, cv=5, scoring='accuracy')
print("\nCross-validation Accuracy (With SMOTE): %.2f (+/- %.2f)" %
      (cv_scores_smote.mean(), cv_scores_smote.std()))

###############################################################################
# 3) Model without Fairness
###############################################################################
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y_encoded)

X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(
    X_ros, y_ros, test_size=0.2, random_state=1
)

scaler_ros = StandardScaler()
X_train_ros[numerical_cols] = scaler_ros.fit_transform(X_train_ros[numerical_cols])
X_test_ros[numerical_cols] = scaler_ros.transform(X_test_ros[numerical_cols])

gbclf_ros = GradientBoostingClassifier(random_state=42)
grid_search_ros = GridSearchCV(estimator=gbclf_ros,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=5, n_jobs=-1)
grid_search_ros.fit(X_train_ros, y_train_ros)

best_gbclf_ros = grid_search_ros.best_estimator_
y_pred_ros = best_gbclf_ros.predict(X_test_ros)
accuracy_ros = accuracy_score(y_test_ros, y_pred_ros)

print("\n=== GradientBoostingClassifier (Random Oversampling) ===")
print("Accuracy:", accuracy_ros)
print("Best Parameters:", grid_search_ros.best_params_)
print("\nClassification Report:")
print(classification_report(y_test_ros, y_pred_ros))
print("Confusion Matrix:")
print(confusion_matrix(y_test_ros, y_pred_ros))

cv_scores_ros = cross_val_score(best_gbclf_ros, X_ros, y_ros, cv=5, scoring='accuracy')
print("\nCross-validation Accuracy (Random Oversampling): %.2f (+/- %.2f)" %
      (cv_scores_ros.mean(), cv_scores_ros.std()))

###############################################################################
# 4) 정책 반영 성능 평균 하락률
###############################################################################
policy_drop_ratio = (accuracy_ros - accuracy_smote) / accuracy_ros * 100
print(f"\n정책 반영 성능 평균 하락률 (Test set): {policy_drop_ratio:.2f}%")
policy_drop_ratio_cv = (cv_scores_ros.mean() - cv_scores_smote.mean()) / cv_scores_ros.mean() * 100
print(f"\n정책 반영 성능 평균 하락률 (CV): {policy_drop_ratio_cv:.2f}%")

###############################################################################
# 5) Compute DEO on the SMOTE model
###############################################################################
# We assume 'Sex' is the protected attribute.
# We'll treat "male" as a=1, "female" as a=-1.
###############################################################################
sex_encoder = categorical_encoders['Sex']
classes_sex = list(sex_encoder.classes_)
male_index = classes_sex.index('male')
female_index = classes_sex.index('female')

test_predictions_smote = best_gbclf_smote.predict(X_test_smote)

# a=1 if male, -1 if female
a_smote = X_test_smote['Sex'].apply(lambda x: 1 if x == male_index else -1)

# TPR_female = P(pred=1 | actual=1, sex=female)
# TPR_male   = P(pred=1 | actual=1, sex=male)
good_mask = (y_test_smote == 1)  # actual=good
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

###############################################################################
# 6) Save Artifacts
###############################################################################
# Save the SMOTE-based (fairness) model
joblib.dump(best_gbclf_smote, 'best_gbclf_smote.joblib')

# Save the base model
joblib.dump(best_gbclf_ros, 'best_gbclf_ros.joblib')

# Save label encoder, categorical encoders, and the SMOTE scaler
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(categorical_encoders, 'categorical_encoders.joblib')
joblib.dump(scaler, 'scaler_smote.joblib')  
joblib.dump(scaler_ros, 'scaler_ros.joblib') 

# Save the lists of numerical and categorical columns
joblib.dump(numerical_cols, 'numerical_cols.joblib')
joblib.dump(categorical_cols, 'categorical_cols.joblib')

print("\nAll artifacts saved. End of training script.")
