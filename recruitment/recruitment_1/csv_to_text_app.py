import pandas as pd
from xlsx2csv import Xlsx2csv
from io import StringIO
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
# import torch

########################이력사항#######################

# GPU 설정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 엑셀 파일 읽기

file_path = '../DATA/applicant.csv'
app_df = pd.read_csv(file_path,low_memory=False)

# 6번째 열과 15번째 열의 데이터 가져오기
columns_to_fetch = [0, 1, 5, 14,37,49] 
selected_data = app_df.iloc[0:10, :] #-> 실제로 할 때는 행 지정 없이

# 출력 형식 변경 및 출력
# 이력서번호: 1열, 지원자번호: 2열
# '분야(field)'에서 '업무(task)'의 업무를 담당
# 분야(field):  6열:세부전공1-> 5열:희망직종1-> 38열:전공
# 업무(task) :  15열:직무태그-> 50열:담당업무(요약모델써서 요약하면됨)

for index, row in tqdm(selected_data.iterrows(), total=len(selected_data), desc="Processing rows"):
    #1번째 열의 데이터 확인
    resume=row.iloc[0] #1번째 열 데이터
    applicant=row.iloc[1] #2번째 열 데이터
    

    # 6번째 열의 데이터 확인
    if pd.isna(row.iloc[5]):  # 6번째 열 데이터가 없는 경우
        if pd.isna(row.iloc[4]):  # 5번째 열 데이터도 없는 경우
            field = row.iloc[37]  # 38번째 열 데이터
        else:
            field = row.iloc[4]  # 5번째 열 데이터
    else:
        field = row.iloc[5]  # 6번째 열 데이터
        
        
    # 15번째 열의 데이터 확인
    if pd.isna(row.iloc[14]):  # 15번째 열 데이터가 없는 경우
        task = row.iloc[49]  # 50번째 열 데이터
    else:
        task = row.iloc[14]  # 15번째 열 데이터

    # field와 task가 모두 존재할 경우만 출력
    if not pd.isna(field) and not pd.isna(task):
        print(f"이력서번호:{resume},지원자번호{applicant},{field}분야에서 {task}의 업무를 담당")

################################################################

# 출력 형식 변경 및 출력
# 이력서번호: 1열, 지원자번호: 2열
# '분야(field)'에서 '업무(task)'의 업무를 담당
# 분야(field):  6열:세부전공1-> 5열:희망직종1-> 38열:전공
# 업무(task) :  15열:직무태그-> 50열:담당업무(요약모델써서 요약하면됨)
######################## 이력사항 #######################
# CSV 파일 읽기
file_path = '../DATA/applicant.csv'
app_df = pd.read_csv(file_path, low_memory=False)

# NLTK와 모델 다운로드
nltk.download('punkt')
model = AutoModelForSeq2SeqLM.from_pretrained('eenzeenee/t5-base-korean-summarization')
# model.to(device)
tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization')


# 6번째 열과 15번째 열의 데이터 가져오기
columns_to_fetch = [0, 1, 5, 14, 37, 49] 
selected_data = app_df.iloc[:, :]  # 실제로 사용할 때는 행 지정 없이

# 저장할 데이터를 담을 리스트
output_data = []

# 요약 함수
def summarize_text(text, prefix="summarize: ", model=model, tokenizer=tokenizer):
    """텍스트를 요약하는 함수"""
    if pd.isna(text) or not text.strip():  # 텍스트가 없는 경우 빈 문자열 반환
        return None
    inputs = tokenizer(prefix + text, max_length=512, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=10, max_length=64)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    if len(nltk.sent_tokenize(decoded_output.strip()))==0 :
        return None
    return nltk.sent_tokenize(decoded_output.strip())[0]
    


# 데이터 처리 및 저장 (TQDM 추가)
for index, row in tqdm(selected_data.iterrows(), total=len(selected_data), desc="Processing rows"):
    # 1번째 열의 데이터 확인
    resume = row.iloc[0]  # 1번째 열 데이터
    applicant = row.iloc[1]  # 2번째 열 데이터

    # 6번째 열의 데이터 확인
    if pd.isna(row.iloc[5]):  # 6번째 열 데이터가 없는 경우
        if pd.isna(row.iloc[4]):  # 5번째 열 데이터도 없는 경우
            field = row.iloc[37]  # 38번째 열 데이터
        else:
            field = row.iloc[4]  # 5번째 열 데이터
    else:
        field = row.iloc[5]  # 6번째 열 데이터

    # 15번째 열의 데이터 확인
    if pd.isna(row.iloc[14]):  # 15번째 열 데이터가 없는 경우
        if pd.isna(row.iloc[49]):
            continue
        task = summarize_text(row.iloc[49])  # 50번째 열 데이터
        
        
    else:
        task = row.iloc[14]  # 15번째 열 데이터


 ##test off
    # field와 task가 모두 존재할 경우만 처리
    if not pd.isna(field) and not pd.isna(task):
        # 3개의 컬럼 데이터로 저장
        output_data.append({
            "이력서번호": resume,
            "지원자번호": applicant,
            "설명": f"{field}분야에서 {task}의 업무를 담당"
        })
        
 ##test on       
    # field와 task가 모두 존재할 경우만 출력
    # if not pd.isna(field) and not pd.isna(task):
    #     print(f"이력서번호:{resume},지원자번호{applicant},{field}분야에서 {task}의 업무를 담당")

# DataFrame으로 변환
output_df = pd.DataFrame(output_data)




# CSV로 저장
output_csv_path = '../DATA/txt_applicant.csv'
output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"CSV 파일로 저장되었습니다: {output_csv_path}")