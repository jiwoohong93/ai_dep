import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import numpy as np

# Configuration
PREDICTIONS_FILE = "./log/medisurveyboost/predictions_20241230_075803.xlsx"
LOG_DIR = "./log/medisurveyboost"

# Create necessary directories
os.makedirs(LOG_DIR, exist_ok=True)

# Load predictions data
print("Loading predictions data...")
data = pd.read_excel(PREDICTIONS_FILE)

# Prepare log file
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"task2_analysis_{current_time}.txt")
with open(log_path, 'w') as log_file:
    def log(message):
        print(message)
        log_file.write(message + "\n")

    # Analyze ground truth
    log("Ground Truth Analysis")

    # Total samples
    total_samples = len(data)
    log(f"Total Samples: {total_samples}")

    # Split by gender
    gender_split = data['Screening_1_인구통계학적 정보_성별'].value_counts()
    log("\nSamples by Gender:")
    log(gender_split.to_string())

    # Split by age groups
    age_bins = [0, 20, 40, 60, 80, float('inf')]
    age_labels = ['-20', '21-40', '41-60', '61-80', '81+']
    data['Age Group'] = pd.cut(data['Screening_1_인구통계학적 정보_나이'], bins=age_bins, labels=age_labels, right=False)
    age_split = data['Age Group'].value_counts()
    log("\nSamples by Age Group:")
    log(age_split.to_string())

    # Split by target classes
    target_split = data['ground_truth'].value_counts()
    log("\nSamples by Target Classes:")
    log(target_split.to_string())

    # Split by 투석 및 이식
    dialysis_groups = [
        '투석 및 이식_22_투석 및 이식_혈액투석여부',
        '투석 및 이식_22_투석 및 이식_복막투석여부',
        '투석 및 이식_22_투석 및 이식_신장이식여부'
    ]

    # Define a new column for "none of the above"
    data['투석 및 이식_22_투석 및 이식_없음'] = (data[dialysis_groups].sum(axis=1) == 0).astype(int)

    dialysis_categories = dialysis_groups + ['투석 및 이식_22_투석 및 이식_없음']

    for category in dialysis_categories:
        group_data = data[data[category] == 1]
        group_samples = len(group_data)
        log(f"\nDialysis Category: {category}")
        log(f"Total Samples: {group_samples}")

        # Split by gender within category
        gender_split_group = group_data['Screening_1_인구통계학적 정보_성별'].value_counts()
        log("Samples by Gender:")
        log(gender_split_group.to_string())

        # Split by age group within category
        age_split_group = group_data['Age Group'].value_counts()
        log("Samples by Age Group:")
        log(age_split_group.to_string())

        # Split by target classes within category
        target_split_group = group_data['ground_truth'].value_counts()
        log("Samples by Target Classes:")
        log(target_split_group.to_string())

    # Analyze predictions
    log("\nPrediction Analysis")
    for category in dialysis_categories:
        group_data = data[data[category] == 1]
        log(f"\nDialysis Category: {category}")

        # Split predictions by target classes
        prediction_split_group = group_data['prediction'].value_counts()
        log("Predictions by Target Classes:")
        log(prediction_split_group.to_string())

    def calculate_consistency(data, feature_columns, prediction_column):
        # Calculate pairwise cosine similarity for features
        feature_matrix = data[feature_columns].values
        similarities = cosine_similarity(feature_matrix)

        # Calculate consistency for each pair
        predictions = data[prediction_column].values
        consistency_scores = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                if similarities[i, j] > 0.9:  # Define a threshold for "similar inputs"
                    consistency = int(predictions[i] == predictions[j])
                    consistency_scores.append(consistency)

        # Return average consistency score
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0

    # Split by randomization groups
    randomization_groups = [
        'Screening_13_무작위 배정_무작위 배정군_집중 공동의사결정군',
        'Screening_13_무작위 배정_무작위 배정군_고전적 대조군',
        'Screening_13_무작위 배정_무작위 배정군_공동의사결정군'
    ]

    feature_columns = [col for col in data.columns if col not in ['ground_truth', 'prediction', 'Age Group'] + randomization_groups]

    for group in randomization_groups:
        group_data = data[data[group] == 1]
        group_samples = len(group_data)
        log(f"\nRandomization Group: {group}")
        log(f"Total Samples: {group_samples}")

        # Calculate accuracy
        accuracy = accuracy_score(group_data['ground_truth'], group_data['prediction'])
        log(f"Accuracy: {accuracy:.4f}")

        # Calculate demographic parity for each class
        log("Demographic Parity for Each Class:")
        for cls in sorted(data['ground_truth'].unique()):
            class_positive_rate = len(group_data[group_data['prediction'] == cls]) / len(group_data)
            log(f"  Class {cls}: {class_positive_rate:.4f}")

        # Calculate confusion matrix
        cm = confusion_matrix(group_data['ground_truth'], group_data['prediction'])
        log("Confusion Matrix:")
        log(np.array2string(cm))

    # Calculate overall consistency score
    log("\nCalculating Consistency Score:")
    consistency = calculate_consistency(data, feature_columns, 'prediction')
    log(f"Consistency Score: {consistency:.4f}")

    log(f"Analysis saved to: {log_path}")
