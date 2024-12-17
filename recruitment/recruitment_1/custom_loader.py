import pandas as pd
import torch
from torch.utils.data import Dataset

import logging
import random
import numpy as np

class ResumeJobDataset(Dataset):
    def __init__(self, sent_transform, 
                 neg_ratio,
                 gt_path, 
                 app_path, 
                 ent_path
                 trainable):
        
        self.train= trainable
        self.sent_transform = sent_transform
        self.negative_ratio = neg_ratio
        
        self.gt = pd.read_csv(gt_path, index_col=0)
        
        self.applicants = pd.read_parquet(app_path)
        self.applicants = self.applicants.drop_duplicates(subset=['이력서번호'], keep='first')
        self.applicants = self.applicants.dropna(subset=['new_label'])
        
        self.enterprises = pd.read_parquet(ent_path)
        self.enterprises = self.enterprises.drop_duplicates(subset=['공고번호'], keep='first')
        
        self.applicant_dict = self.applicants.set_index('이력서번호').to_dict('index')
        self.enterprise_dict = self.enterprises.set_index('공고번호').to_dict('index')
        self.positive_pairs = self.gt[['이력서번호', '공고번호']].dropna().values.tolist()
        
        self.bias_sent = ["남성입니다", "여성입니다"]
        #self.bias_embed = self.sent_transform.encode(self.bias_sent) 
        self.bias_embed = [np.ones(768*2, dtype=np.float32) * 10, -10 * np.ones(768*2, dtype=np.float32)]
        
        # 모든 이력서와 공고 ID 리스트
        self.all_applicant_ids = self.applicants['이력서번호'].tolist()
        self.all_job_ids = self.enterprises['공고번호'].tolist()
        
        # 매칭된 쌍을 set으로 저장하여 부정 샘플을 생성할 때 참고
        self.matched_set = set(tuple(x) for x in self.positive_pairs) # (())
                
        self.filtered_positive_pairs = []
        self.missing_resumes = set()
        for pair in self.positive_pairs:
            resume_id, job_id = pair
            if resume_id in self.applicant_dict and job_id in self.enterprise_dict:
                self.filtered_positive_pairs.append(pair)
            else:
                self.missing_resumes.add(resume_id)

        # 로깅: 누락된 이력서번호 보고
        if self.missing_resumes:
            logging.warning(f"{len(self.missing_resumes)}개의 이력서번호가 applicants 데이터에 존재하지 않습니다.")
            # for resume_id in self.missing_resumes:
            #     logging.warning(f"누락된 이력서번호: {resume_id}")

        # 긍정 샘플 수 업데이트
        self.positive_pairs = self.filtered_positive_pairs
        logging.info(f"필터링 후 긍정 샘플 수: {len(self.positive_pairs)}")
        
        # 샘플 생성
        self.samples = []
        for pair in self.positive_pairs:
            self.samples.append((pair[0], pair[1], 1))  # 긍정 샘플
            # 부정 샘플 생성
            for _ in range(self.negative_ratio):
                negative_job_id = random.choice(self.all_job_ids)
                # 매칭되지 않은 경우만 추가
                while (pair[0], negative_job_id) in self.matched_set:
                    negative_job_id = random.choice(self.all_job_ids)
                self.samples.append((pair[0], negative_job_id, 0)) 
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        resume_id, job_id, label = self.samples[idx]
        resume_embed = self.applicant_dict[resume_id]['embedding']
        job_embed = self.enterprise_dict[job_id]['embedding']
        
        new_label = self.applicant_dict[resume_id]['new_label']
        
        if self.train
            bias_embed = self.bias_embed[int(new_label)]
        else:
            bias_embed = self.bias_embed[1- int(new_label)]
        
        new_resume_embed = torch.from_numpy(np.hstack((resume_embed, bias_embed)))
        #new_resume_embed = torch.from_numpy(resume_embed +bias_embed)
        job_embed = torch.from_numpy(job_embed)
        label = torch.tensor(label)

        return resume_id, new_resume_embed, job_embed, label

class CustomResumeJobDataset(Dataset):
    def __init__(self, samples, applicant_dict, enterprise_dict, bias_embed, trainable):
        self.samples = samples
        self.applicant_dict = applicant_dict
        self.enterprise_dict = enterprise_dict
        self.bias_embed = bias_embed
        self.train = trainable
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        resume_id, job_id, label = self.samples[idx]
        resume_embed = self.applicant_dict[resume_id]['embedding']
        job_embed = self.enterprise_dict[job_id]['embedding']
        
        new_label = self.applicant_dict[resume_id]['new_label']
        
        if self.train
            bias_embed = self.bias_embed[int(new_label)]
        else:
            bias_embed = self.bias_embed[1- int(new_label)]
        new_resume_embed = torch.from_numpy(np.hstack((resume_embed, bias_embed)))
        #new_resume_embed = torch.from_numpy(resume_embed +bias_embed)
        job_embed = torch.from_numpy(job_embed)
        label = torch.tensor(label)

        return resume_id, new_resume_embed, job_embed, label



if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    ds = ResumeJobDataset(sent_transform = model, 
                neg_ratio = 1,
                gt_path = "../DATA/gt.csv", 
                app_path = "../Modeling/txt_applicant_updated.parquet", 
                ent_path = "../Modeling/txt_enterprise.parquet")
    import ipdb;ipdb.set_trace()