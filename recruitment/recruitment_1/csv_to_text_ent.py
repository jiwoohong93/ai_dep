import pandas as pd
import torch
####################### 공고내역 ##############################
# 엑셀 파일 읽기 (두 번째 시트 지정)
file_path = '../DATA/enterprise3.csv'
ent_df = pd.read_csv(file_path,low_memory=False)  # 두 번째 시트는 0-based index로 1




# 29번째 열과 8번째 열의 데이터 가져오기
selected_data = ent_df.iloc[0:10, :]  # 실제로 사용할 때는 행 지정 없이

# 출력 형식 변경 및 출력
# 기업번호: 2열, 공고번호: 3열
# '분야(field)'에서 '업무(task)'의 업무를 담당
# 분야(field):  29열: 키워드 -> 4열: 공고명
# 업무(task):  8열: 세부직종1
for index, row in selected_data.iterrows():
    #2번째 열의 데이터 확인
    ent=row.iloc[1]
    
    #3번째 열의 데이터 확인
    job=row.iloc[2]
    
    # 29번째 열의 데이터 확인
    if pd.isna(row.iloc[28]):  # 29번째 열 데이터가 없는 경우
        field = row.iloc[3]  # 4번째 열 데이터
    else:
        field = row.iloc[28]  # 29번째 열 데이터

    # 8번째 열의 데이터 확인
    task = row.iloc[7]  # 8번째 열 데이터

    # field와 task가 모두 존재할 경우만 출력
    if not pd.isna(field) and not pd.isna(task):
        print(f"기업번호:{ent}, 공고번호:{job}, {field}분야에서 {task}의 업무를 담당")

##################################################################################################
import pandas as pd

####################### 공고내역 ##############################
# CSV 파일 읽기
file_path = '../DATA/enterprise3.csv'
ent_df = pd.read_csv(file_path, low_memory=False)

# 29번째 열과 8번째 열의 데이터 가져오기
selected_data = ent_df.iloc[:, :]  # 실제로 사용할 때는 행 지정 없이

# 저장할 데이터를 담을 리스트
output_data = []

# 데이터 처리 및 저장
for index, row in selected_data.iterrows():
    # 2번째 열의 데이터 확인
    ent = row.iloc[1]

    # 3번째 열의 데이터 확인
    job = row.iloc[2]

    # 29번째 열의 데이터 확인
    if pd.isna(row.iloc[28]):  # 29번째 열 데이터가 없는 경우
        field = row.iloc[3]  # 4번째 열 데이터
    else:
        field = row.iloc[28]  # 29번째 열 데이터

    # 8번째 열의 데이터 확인
    task = row.iloc[7]  # 8번째 열 데이터

    # field와 task가 모두 존재할 경우만 처리
    if not pd.isna(field) and not pd.isna(task):
        # 3개의 컬럼 데이터로 저장
        output_data.append({
            "기업번호": ent,
            "공고번호": job,
            "설명": f"{field}분야에서 {task}의 업무를 담당"
        })

# DataFrame으로 변환
output_df = pd.DataFrame(output_data)

# CSV로 저장
output_csv_path = '../DATA/txt_enterprise.csv'
output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"CSV 파일로 저장되었습니다: {output_csv_path}")
