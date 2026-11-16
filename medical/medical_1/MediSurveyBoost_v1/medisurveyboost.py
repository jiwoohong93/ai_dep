import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from dataset import task2_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import numpy as np

# Configuration
CSV_FILE = "./SDM-ART/p_SDM-ART_20241129.csv"
LOG_DIR = "./log/medisurveyboost"

PROTECTED_FEATURE = 'Visit 1_14_설문지_교육 후 자가진단도구_26. 나는 돌봐야 할 노부모와 함께 살고 있다.'
PROTECTED_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

rs = 43

# Create necessary directories
os.makedirs(LOG_DIR, exist_ok=True)

# 1. Preprocess Dataset
print("Preprocessing the dataset...")
full_dataset = task2_dataset(csv_file=CSV_FILE) 
features, targets = full_dataset.features.values, full_dataset.targets  # Use `values` for numpy array
feature_columns = full_dataset.feature_columns  # Get original column names

# Combine features and targets into a single DataFrame
data = pd.DataFrame(features, columns=feature_columns)
data['target'] = targets

# 2. Cross-Validation Setup
n_splits = 5  # Number of folds
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)

# Initialize XGBoost model
model = XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.02,
    n_estimators=500,
    max_depth=4,
    random_state=rs,
    reg_lambda=2.0
)

# Cross-Validation Loop
fold_accuracies = []
all_predictions = []  # To store predictions for all folds

print("Starting Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(data.drop(columns=['target']), data['target'])):
    print(f"\nFold {fold + 1}/{n_splits}")
    
    # Split data into train and validation for this fold
    X_train, X_val = data.drop(columns=['target']).iloc[train_idx], data.drop(columns=['target']).iloc[val_idx]
    y_train, y_val = data['target'].iloc[train_idx], data['target'].iloc[val_idx]
    
    # Train model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Apply causal intervention during inference
    adjusted_predictions = []
    for idx, row in X_val.iterrows():
        intervention_results = []
        for value in PROTECTED_VALUES:
            row_copy = row.copy()
            row_copy[PROTECTED_FEATURE] = value
            row_copy = row_copy.values.reshape(1, -1)  # Reshape for prediction
            intervention_results.append(model.predict_proba(row_copy)[0][1])  # Probability of positive class
        
        # Weighted average of predictions
        adjusted_prediction = np.mean(intervention_results)
        adjusted_predictions.append(1 if adjusted_prediction >= 0.5 else 0)
    
    # Evaluate model
    fold_accuracy = accuracy_score(y_val, adjusted_predictions)
    fold_accuracies.append(fold_accuracy)
    
    print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
    
    # Store predictions with inputs and ground truth
    fold_df = X_val.copy()  # Copy validation input features
    fold_df['ground_truth'] = y_val.values
    fold_df['prediction'] = adjusted_predictions
    fold_df['fold'] = fold + 1  # Add fold identifier
    all_predictions.append(fold_df)

# Combine all fold predictions
all_predictions_df = pd.concat(all_predictions, ignore_index=True)

# Save predictions to Excel
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
predictions_path = os.path.join(LOG_DIR, f"predictions_{current_time}.xlsx")
all_predictions_df.to_excel(predictions_path, index=False)
print(f"Predictions saved to: {predictions_path}")

# Compute Average Accuracy
average_accuracy = np.mean(fold_accuracies)
print(f"\nCross-Validation Average Accuracy: {average_accuracy:.4f}")

# 3. Feature Importance
print("Analyzing feature importance...")
importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display top 20 features
top_20_features = importance_df.head(20)
print("\nTop 20 Features by Importance:")
print(top_20_features)

# Display bottom 20 features (least important)
bottom_20_features = importance_df.tail(20).sort_values(by='Importance', ascending=True)
print("\nBottom 20 Features by Importance:")
print(bottom_20_features)

# Save top 20 and bottom 20 features to text files
top_20_path = os.path.join(LOG_DIR, f"top_20_features_{current_time}.txt")
bottom_20_path = os.path.join(LOG_DIR, f"bottom_20_features_{current_time}.txt")

with open(top_20_path, 'w') as f:
    f.write("Top 20 Features by Importance:\n")
    f.write(top_20_features.to_string(index=False))

with open(bottom_20_path, 'w') as f:
    f.write("Bottom 20 Features by Importance:\n")
    f.write(bottom_20_features.to_string(index=False))

print(f"Top 20 features saved to: {top_20_path}")
print(f"Bottom 20 features saved to: {bottom_20_path}")

# 4. Analyze Effects of Top 20 Features on the Target
analysis_log_path = os.path.join(LOG_DIR, f"feature_effect_analysis_{current_time}.txt")
with open(analysis_log_path, 'w') as f:
    f.write("Feature Effect Analysis:\n")
    for feature in top_20_features['Feature']:
        if feature in data.columns:
            f.write(f"\nFeature: {feature}\n")
            grouped = data.groupby(feature)['target'].mean()
            f.write("Target Mean by Feature Value:\n")
            f.write(grouped.to_string())
            f.write("\n")
print(f"Feature effect analysis saved to: {analysis_log_path}")

# 5. Save the Model
task_log_dir = os.path.join(LOG_DIR, current_time)
os.makedirs(task_log_dir, exist_ok=True)

best_model_path = os.path.join(task_log_dir, "medisurveyboost_crossval_model.xgb")
model.save_model(best_model_path)
print(f"Best model saved to: {best_model_path}")
