import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
from tqdm import tqdm

########################################
# 1) CONFIGURATION
########################################

TEST_LABELS_CSV = "./data/four_findings_expert_labels_test_labels.csv"
VAL_LABELS_CSV  = "./data/four_findings_expert_labels_validation_labels.csv"

IMG_DIRS = [
    "./data/datasets/nih-chest-xrays/data/versions/3/images_001/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_002/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_003/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_004/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_005/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_006/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_007/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_008/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_009/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_010/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_011/images",
    "./data/datasets/nih-chest-xrays/data/versions/3/images_012/images",
]

EPOCHS = 2
BATCH_SIZE = 8
LR = 5e-5
TEST_EVERY_N_STEPS = 100 * 8 // BATCH_SIZE
SPLIT_RATIO = 0.10 
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

########################################
# 2) UTILITY FUNCTIONS
########################################

def find_image_path(img_name: str):
    for d in IMG_DIRS:
        fullp = os.path.join(d, img_name)
        if os.path.exists(fullp):
            return fullp
    return None

def get_gender_label(gender_str: str):
    if gender_str.strip().upper() == "M":
        return 1
    else:
        return -1

########################################
# 3) READ & MERGE THE TWO CSVs
########################################

def merge_expert_csvs(csv_file1, csv_file2):
    merged_rows = []
    # read first CSV
    with open(csv_file1, 'r') as f1:
        r1 = csv.DictReader(f1)
        for row in r1:
            merged_rows.append(row)
    # read second CSV
    with open(csv_file2, 'r') as f2:
        r2 = csv.DictReader(f2)
        for row in r2:
            merged_rows.append(row)
    return merged_rows

########################################
# 4) DATASET DEFINITION
########################################

class CleanChestDataset(Dataset):
    def __init__(self, all_rows, transform=None, indices=None):
        super().__init__()
        self.transform = transform
        self.samples = []

        print(f"[Dataset Init] Reading {len(all_rows)} total rows from merged CSV...")

        missing_img_count = 0
        age_continued = 0
        for i in range(len(all_rows)):
            row = all_rows[i]
            img_name = row["Image Index"].strip()
            full_path = find_image_path(img_name)
            if not full_path:
                missing_img_count += 1
                continue

            is_pneumothorax = row['Pneumothorax'] == 'YES'

            if is_pneumothorax:
                cond_label = 0
            else:
                cond_label = 1

            # Protected attribute (gender)
            gval = get_gender_label(row["Patient Gender"])

            self.samples.append((full_path, cond_label, gval))

        print(f"[Dataset Init] Dropped {missing_img_count} because images not found.")
        print(f"[Dataset Init] Total samples so far: {len(self.samples)}")

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
            print(f"[Dataset Init] After subsetting => {len(self.samples)} samples remain.")

        # Print label distribution
        label_counts = Counter(s[1] for s in self.samples)
        print("[Dataset Init] Label distribution:")
        for lv, ccount in label_counts.items():
            lname = "No Finding (0)" if lv == 0 else "Positive (1)"
            print(f"  {lname}: {ccount}")

        # Gender distribution
        gender_counts = Counter(s[2] for s in self.samples)
        print("[Dataset Init] Gender distribution:")
        for gval, gcount in gender_counts.items():
            gname = "Male(+1)" if gval == 1 else "Female(-1)"
            print(f"  {gname}: {gcount}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cond_label, gender_label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, cond_label, gender_label

    def get_metadata(self):
        return [(s[1], s[2]) for s in self.samples]

########################################
# 5) MODEL DEFINITION
########################################

def initialize_model(num_classes=2, use_pretrained=True):
    model = models.resnet18(pretrained=use_pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

########################################
# 6) EVALUATION
########################################

def evaluate_DEO(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_gender = []

    with torch.no_grad():
        for images, labels, gender in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gender.extend(gender.numpy())

    acc = accuracy_score(all_labels, all_preds)

    # TPR_f vs TPR_m
    female_idx = [i for i, g in enumerate(all_gender) if g == -1]
    male_idx   = [i for i, g in enumerate(all_gender) if g == 1]

    female_actual_1 = [i for i in female_idx if all_labels[i] == 1]
    male_actual_1   = [i for i in male_idx   if all_labels[i] == 1]

    if len(female_actual_1) > 0:
        female_pred_1 = sum(all_preds[j] == 1 for j in female_actual_1)
        tpr_f = female_pred_1 / len(female_actual_1)
    else:
        tpr_f = 0.0

    if len(male_actual_1) > 0:
        male_pred_1 = sum(all_preds[j] == 1 for j in male_actual_1)
        tpr_m = male_pred_1 / len(male_actual_1)
    else:
        tpr_m = 0.0

    DEO = abs(tpr_f - tpr_m)
    return acc, DEO, tpr_f, tpr_m

########################################
# 7) TRAINING HELPER
########################################

def train_one_epoch(model, loader, optimizer, criterion, device,
                    test_loader, step_counter,
                    eval_interval=50):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for i, (images, labels, gender) in tqdm(enumerate(loader)):
        step_counter += 1
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        if step_counter % eval_interval == 0:
            print(f"[Step {step_counter}] Running test evaluation..")
            test_acc, test_deo, tpr_f, tpr_m = evaluate_DEO(model, test_loader, device)
            print(f"[Step {step_counter}] Test Accuracy={test_acc:.4f}, DEO={test_deo:.4f}, "
                  f"TPR_f={tpr_f:.3f}, TPR_m={tpr_m:.3f}")

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc, step_counter

########################################
# 8) FULL TRAINING PIPELINE
########################################

def full_training_procedure(train_dataset, test_dataset,
                            use_fairness=False,
                            epochs=1,
                            batch_size=8,
                            lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(num_classes=2, use_pretrained=True).to(device)

    group_dict = defaultdict(list)
    train_labels = []

    meta = train_dataset.get_metadata()
    for idx, (c_label, g_label) in enumerate(meta):
        group_dict[(c_label, g_label)].append(idx)
        train_labels.append(c_label)

    if use_fairness:
        print("[With Fairness] WeightedRandomSampler across subgroups..")
        group_sizes = {g: len(idxs) for g, idxs in group_dict.items()}
        max_size = max(group_sizes.values()) if group_sizes else 1
        sample_weights = np.zeros(len(meta), dtype=np.float32)
        for g, idxs in group_dict.items():
            sg_size = len(idxs)
            if sg_size > 0:
                w = float(max_size / sg_size)
            else:
                w = 1.0
            for i in idxs:
                sample_weights[i] = w
        sampler = WeightedRandomSampler(sample_weights, len(meta))
        shuffle = False
    else:
        print("[No Fairness] Standard shuffle.")
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, shuffle=shuffle, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    cnt = Counter(train_labels)
    print("[Training] Label counts => 0:", cnt[0], " 1:", cnt[1])
    tot = cnt[0] + cnt[1] if (cnt[0]+cnt[1]) else 1
    w0 = cnt[1]/tot
    w1 = cnt[0]/tot
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
    class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0
    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}] Starting..")
        ep_loss, ep_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            test_loader=test_loader,
            step_counter=global_step,
            eval_interval=TEST_EVERY_N_STEPS
        )
        print(f"[Epoch {epoch+1}] Done => Loss={ep_loss:.4f}, Acc={ep_acc:.4f}")

    print("\n*** Final Test Eval ***")
    f_acc, f_deo, tpr_f, tpr_m = evaluate_DEO(model, test_loader, device)
    print(f"[Test] Acc={f_acc:.4f}, DEO={f_deo:.4f}, TPR_f={tpr_f:.3f}, TPR_m={tpr_m:.3f}")

    return f_acc, f_deo, model

########################################
# 9) MAIN
########################################

def main():
    print("[Main] Merging new CSVs..")
    merged_rows = merge_expert_csvs(TEST_LABELS_CSV, VAL_LABELS_CSV)
    print(f"[Main] Merged total rows = {len(merged_rows)}")

    full_dataset = CleanChestDataset(merged_rows, transform=None)

    if len(full_dataset) == 0:
        print("No data after merging. Exiting.")
        return

    all_indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=SPLIT_RATIO, random_state=SEED, shuffle=True
    )
    print(f"[Main] Train indices => {len(train_idx)}, Test indices => {len(test_idx)}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.81,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset = CleanChestDataset(merged_rows, transform=train_transform, indices=train_idx)
    test_dataset  = CleanChestDataset(merged_rows, transform=test_transform,  indices=test_idx)

    print("\n======================")
    print("MODEL B (WITH FAIRNESS)")
    print("======================")
    acc_b, deo_b, model_b = full_training_procedure(
        train_dataset, test_dataset,
        use_fairness=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )

    print("\n======================")
    print("MODEL A (NO FAIRNESS)")
    print("======================")
    acc_a, deo_a, model_a = full_training_procedure(
        train_dataset, test_dataset,
        use_fairness=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )

    # 6) Compare
    if acc_a > 0:
        drop_ratio = ((acc_a - acc_b) / acc_a)*100
    else:
        drop_ratio = 0
    print("\n*** FINAL COMPARISON ***")
    print(f"[No Fairness]   Acc={acc_a:.4f}, DEO={deo_a:.4f}")
    print(f"[With Fairness] Acc={acc_b:.4f}, DEO={deo_b:.4f}")
    print(f"정책 반영 성능 평균 하락률 = {drop_ratio:.2f}%")

    # 7) Save
    torch.save(model_a.state_dict(), "model_no_fairness.pth")
    torch.save(model_b.state_dict(), "model_with_fairness.pth")
    print("[Main] Done. Models saved.")

if __name__ == "__main__":
    main()
