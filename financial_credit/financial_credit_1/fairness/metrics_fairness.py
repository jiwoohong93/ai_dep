import torch
import torch.nn as nn

# Function to compute accuracy
def compute_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for feat_cat, feat_num, labels in data_loader:
            feat_cat = feat_cat.to(device)
            feat_num = feat_num.to(device)
            labels = labels.to(device)

            predicted, _, _ = model(feat_cat, feat_num)
            causal_feat_cat = 1 - feat_cat
            causal_predicted, _, _ = model(causal_feat_cat, feat_num)
            weights = torch.where(feat_cat == 1, 1.52, 2.94)
            causal_weights = torch.where(causal_feat_cat == 1, 1.52, 2.94)
            final_predicted = predicted * weights + causal_predicted * causal_weights
            final_predicted = nn.Sigmoid()(final_predicted)
            predicted = final_predicted.round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Function to calculate Differential Equality of Opportunity (DEO)
def compute_deo(model, data_loader, device):
    model.eval()
    tpr_male = 0
    tpr_female = 0
    total_male = 0
    total_female = 0
    with torch.no_grad():
        for feat_cat, feat_num, labels in data_loader:
            feat_cat = feat_cat.to(device)
            feat_num = feat_num.to(device)
            labels = labels.to(device)

            predicted, _, _ = model(feat_cat, feat_num)
            causal_feat_cat = 1 - feat_cat
            causal_predicted, _, _ = model(causal_feat_cat, feat_num)
            weights = torch.where(feat_cat == 1, 1.52, 2.94)
            causal_weights = torch.where(causal_feat_cat == 1, 1.52, 2.94)
            final_predicted = predicted * weights + causal_predicted * causal_weights
            final_predicted = nn.Sigmoid()(final_predicted)
            predictions = final_predicted.round()
            sex = feat_cat
            # sex = feat_cat[:, -2]
            # Male
            male_mask = (sex == 1)
            correct_male = (predictions[male_mask] == labels[male_mask]).float()
            total_male += male_mask.sum().item()
            tpr_male += correct_male.sum().item()
            # Female
            female_mask = (sex == 0)
            correct_female = (predictions[female_mask] == labels[female_mask]).float()
            total_female += female_mask.sum().item()
            tpr_female += correct_female.sum().item()
    tpr_male = tpr_male / total_male if total_male else 0
    tpr_female = tpr_female / total_female if total_female else 0
    return abs(tpr_male - tpr_female)  # DEO is the absolute difference in TPRs
