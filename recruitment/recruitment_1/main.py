import pandas as pd
import random, os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import TwoVectorClassifier
from custom_loader import ResumeJobDataset, CustomResumeJobDataset
from train import train, test
from sentence_transformers import SentenceTransformer

import argparse    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(
         neg_ratio: int = 1,
         gt_path:str = "../DATA/gt.csv", 
         app_path:str = "txt_applicant_updated.parquet",
         ent_path:str = "txt_enterprise.parquet",
         batch_size: int = 64,
         num_epochs: int = 3,
         learning_rate: float = 3e-5,
         patience: int = 3,
         mode: str = "768_2",
         train_ = False
         ):
    
    #TODO: Seed고정
    set_seed(42)
    
    #TODO: 데이터로드
    sent_embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    full_dataset = ResumeJobDataset(sent_embed_model, neg_ratio, gt_path, app_path, ent_path, train_)
    
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    
    samples = full_dataset.samples

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(val_size + test_size),
        random_state=42,
        stratify=[sample[2] for sample in samples]
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(test_size / (val_size + test_size)),
        random_state=42,
        stratify=[sample[2] for sample in temp_samples]
    )
    
    applicant_dict = full_dataset.applicant_dict
    enterprise_dict = full_dataset.enterprise_dict
    bias_embed = full_dataset.bias_embed
    
    # 9. Custom Dataset 인스턴스 생성
    train_dataset = CustomResumeJobDataset(
        samples=train_samples,
        applicant_dict=applicant_dict,
        enterprise_dict=enterprise_dict,
        bias_embed=bias_embed,
        trainable = train_
    )

    val_dataset = CustomResumeJobDataset(
        samples=val_samples,
        applicant_dict=applicant_dict,
        enterprise_dict=enterprise_dict,
        bias_embed=bias_embed,
        trainable = train_
    )

    test_dataset = CustomResumeJobDataset(
        samples=test_samples,
        applicant_dict=applicant_dict,
        enterprise_dict=enterprise_dict,
        bias_embed=bias_embed,
        trainable = train_
    )
    
    #TODO: 모델 정의    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': []
    }
    
    #TODO: 학습!
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = TwoVectorClassifier(input_dim=768, reduced_dim=256, hidden_dim=128)
    sd = torch.load(f"{mode}/model.pth"); model.load_state_dict(sd['state_dict'], strict=True)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    if not os.path.isdir(mode):
        os.mkdir(mode)
        
    best_f1 = 0.0
    if train_:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # 학습
            train_loss = train(model, train_dataloader, optimizer, criterion, device)
            print(f"Training Loss: {train_loss:.4f}")

            # 평가
            metrics, _ = test(model, val_dataloader, criterion, device)
            print(f"Validation Loss: {metrics['loss']:.4f}")
            print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
            print(f"Validation Precision: {metrics['precision']:.4f}")
            print(f"Validation Recall: {metrics['recall']:.4f}")
            print(f"Validation F1 Score: {metrics['f1_score']:.4f}")
            
            # 기록 저장
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(metrics['loss'])
            history['val_accuracy'].append(metrics['accuracy'])
            history['val_precision'].append(metrics['precision'])
            history['val_recall'].append(metrics['recall'])
            history['val_f1_score'].append(metrics['f1_score'])
            
            # 최고 F1 Score를 기록한 모델 저장
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                sd = {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_score": best_f1
                }
                torch.save(sd, f'{mode}/model2.pth')
                print(f"모델 저장 완료 ==> {epoch + 1} with F1 Score: {best_f1:.4f}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("*"*10,"조기 종료", "*"*10)
                    break
        
        # 학습 기록을 DataFrame으로 변환 후 CSV로 저장
        history_df = pd.DataFrame(history)
        history_df.to_csv(f'{mode}/training_history.csv', index=False)
        print("학습 및 검증 단계 저장 완료!")
    
    # 최종 테스트
    sd = torch.load(f"{mode}/model2.pth")
    model.load_state_dict(sd['state_dict'], strict=True)
    print("Testing on Test Dataset")
    model.load_state_dict(sd['state_dict'], strict=True)
    test_metrics, total_info = test(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    
    if train_:
        # 테스트 결과 저장
        test_history = {
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1_score': test_metrics['f1_score']
        }
        
        test_history_df = pd.DataFrame([test_history])
        test_history_df.to_csv(f'{mode}/test_results.csv', index=False)
        print("테스트 결과 저장 완료!")
        
        total_info_df = pd.DataFrame(total_info)
        total_info_df.to_csv(f'{mode}/test_preds.csv', index=False)
    else:
                # 테스트 결과 저장
        test_history = {
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1_score': test_metrics['f1_score']
        }
        
        test_history_df = pd.DataFrame([test_history])
        test_history_df.to_csv(f'{mode}/bias_results.csv', index=False)
        print("테스트 결과 저장 완료!")
        
        total_info_df = pd.DataFrame(total_info)
        total_info_df.to_csv(f'{mode}/bias_preds.csv', index=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_", type=bool, default=False, help="Test Process or not")          
    parser.add_argument("--mode", type=str, help="save directory(folder)") 
    parser.add_argument("--gt_path", type=str, help="dir for ground truth data")          
    parser.add_argument("--app_path", type=str, help="dir for applicant data") 
    parser.add_argument("--ent_path", type=str, help="dir for enterprise data")          
    main()