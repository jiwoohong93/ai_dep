import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import ipdb

class task2_dataset(Dataset):
    #  응급 투석 여부 (binary) : input=screening+visit1+투석정보, output= 응급투석 여부
    def __init__(self, csv_file, target_column='투석 및 이식_23_응급투석_비계획 투석 여부'):
        """
        Args:
            csv_file (str): Path to the CSV file.
            target_column (str): Target column for binary classification.
        """
        # List of substrings to check in column names
        columns_to_drop = ['연구대상자ID', '기관코드', 'Visit 7', 'Post hoc']

        # Dynamically drop columns that contain any substring from the list
        self.data = pd.read_csv(csv_file).drop(
            columns=[col for col in pd.read_csv(csv_file).columns if any(substring in col for substring in columns_to_drop)],
            errors='ignore'
        )

        self.target_column = target_column
        # self.data.to_excel("before_encoded_data.xlsx", index=False)
        # Preprocessing
        self.data = self.encoding(self.data)
        # self.data.to_excel("encoded_data.xlsx", index=False)
        
        self.targets = self.data[self.target_column].values  # Binary target column (1.0 or 0.0)

        # Separate features and target. drop unnecessary info.
        self.features = self.data.drop(columns=self.target_column)#.values
        self.feature_columns = self.features.columns

    def encoding(self, df):       
        """
        Preprocess and encode the DataFrame according to specific rules:
        """
        # Step 1: Handle numeric columns (outlier removal and normalization)
        outlier_values = [8888, 8888.0, 888, 888.0, 88.8, 888.88, 8888888888, 8888888888.0]

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Identify min and max values
                min_val = df[col].min()
                max_val = df[col].max()

                # Replace outliers only if they are min or max
                df[col] = df[col].apply(
                    lambda x: np.nan if x in outlier_values and (x == min_val or x == max_val) else x
                )

                # # Normalize column
                # min_val = df[col].min()  # Recalculate after replacing outliers
                # max_val = df[col].max()
                # if max_val != min_val:  # Prevent division by zero
                #     df[col] = (df[col] - min_val) / (max_val - min_val)


        # Step 2: Map specific yes/no/normal/abnormal values
        value_mapping = {
            '예': 1.0, 'Yes': 1.0, '비정상': 1.0, 'Abnormal': 1.0, 'Male': 1.0, 
            '아니오': 0.0, 'No': 0.0, '정상': 0.0, 'Normal': 0.0, '': 0.0, 'Female': 0.0, 'Unknown': 0.0
        }
        for col in df.columns:
            df[col] = df[col].map(value_mapping).fillna(df[col])  # Map values, leave unmapped values

        # Step 3: Handle '삶의질' columns with specific mappings
        qol_mapping = {
            5.0: ['전혀 지장이 없다', '전혀 통증이나 불편감이 없다', '전혀 불안하거나 우울하지 않다', '어려움이 전혀 없었다', 
                  '나는 전혀 통증이 없었다', '항상 기운이 있었다', '전혀 우울하지 않았다', '항상 행복'],
            4.0: ['약간 지장이 있다', '약간 통증이나 불편감이 있다', '약간 불안하거나 우울하다', '어려움이 약간 있었다',
                  '나는 약한 통증이 있었다', '자주 기운이 있었다', '가끔 우울', '자주 행복'],
            3.0: ['중간 정도의 지장이 있다', '중간 정도의 통증이나 불편감이 있다', '중간 정도로 불안하거나 우울하다'],
            2.0: ['심한 지장이 있다', '나는 심한 통증이나 불편감이 있다', '심하게 불안하거나 우울하다', '어려움이 많이 있었다',
                  '나는 심한 통증이 있었다', '가끔 기운이 있었다', '자주 우울', '가끔 행복'],
            1.0: ['수 없다', '나는 극심한 통증이나 불편감이 있다', '극도로 불안하거나 우울하다', '오를 수 없었다',
                  '나는 극심한 통증이 있었다', '전혀 기운이 없었다', '항상 우울', '전혀 행복하지 않았다']
        }
        for col in df.columns:
            if '삶의질' in col:
                for value, text_list in qol_mapping.items():
                    df[col] = df[col].apply(
                        lambda x: value if isinstance(x, str) and any(text in x for text in text_list) else x
                    )

        # Step 3-2: 
        qol_mapping = {
            5.0: ['대학원'],
            4.0: ['대학교'],
            3.0: ['고등학교'],
            2.0: ['중학교'],
            1.0: ['초등학교'],
            0.0: ['무응답', '무학', '모름']
        }
        for col in df.columns:
            if '최종학력' in col:
                for value, text_list in qol_mapping.items():
                    df[col] = df[col].apply(
                        lambda x: value if isinstance(x, str) and any(text in x for text in text_list) else x
                    )


        # Step 3-2: 
        qol_mapping = {
            5.0: ['완전 의존'],
            4.0: ['<100%'],
            3.0: ['≤ 50%'],
            2.0: ['독립적'],
            1.0: ['무응답']
        }
        for col in df.columns:
            if '인구통계학적' in col:
                for value, text_list in qol_mapping.items():
                    df[col] = df[col].apply(
                        lambda x: value if isinstance(x, str) and any(text in x for text in text_list) else x
                    )


        # Step 4: Handle '비용조사' columns with specific mappings
        cost_mapping = {
            1.0: ['있다', '예'],
            0.0: ['없다', '아니오']
        }
        for col in df.columns:
            if '비용조사' in col:
                for value, text_list in cost_mapping.items():
                    df[col] = df[col].apply(
                        lambda x: value if isinstance(x, str) and any(text in x for text in text_list) else x
                    )

        # Step 5: Fill NaN values with 0.0
        df.fillna(0.0, inplace=True)

        # Step 6: One-hot encode columns with non-numeric values while excluding 0.0
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):  # Check if column has non-numeric values
                # Exclude '0.0' (which replaced NaN) from the one-hot encoding
                unique_values = df[col].dropna().unique()
                unique_values = [val for val in unique_values if val != 0.0]

                for val in unique_values:
                    new_col_name = f"{col}_{val}"
                    df[new_col_name] = df[col].apply(lambda x: 1.0 if x == val else 0.0)

                # Drop the original column
                df.drop(columns=[col], inplace=True)

        return df

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert features and target to tensors
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)  # Binary target for classification
        return features, target
