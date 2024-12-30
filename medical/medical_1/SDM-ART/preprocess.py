import pandas as pd
import numpy as np
import ipdb
from tqdm import tqdm
import os
import ast

# Load the Excel file
file_name = 'SDM-ART_20241129.xlsx'
sheets = pd.ExcelFile(file_name)

# Process the first sheet '1_인구통계학적 정보'
df_first = sheets.parse('1_인구통계학적 정보')

# Extract '연구대상자ID' and '기관코드' columns
id_column = df_first['연구대상자ID']
institution_code_column = df_first['기관코드']

# Initialize a DataFrame to store combined results
combined_df = pd.DataFrame({
    '연구대상자ID': id_column,
    '기관코드': institution_code_column
})
num = combined_df.shape[0]

# Iterate through all sheets with a progress bar
for sheet_name in tqdm(sheets.sheet_names, desc="Processing Sheets"):
    print(f"Processing sheet: {sheet_name}")
    
    # Read the current sheet
    df = sheets.parse(sheet_name)

    # Ensure '연구대상자ID' and 'Visit명' columns are present
    if '연구대상자ID' not in df.columns or 'Visit명' not in df.columns:
        ipdb.set_trace()
        continue

    # Check for unique 'Visit명' values
    unique_visits = df['Visit명'].dropna().unique()
    print(f"Unique 'Visit명' values in {sheet_name}: {unique_visits}")

    # Iterate through unique Visit명 values
    for visit_name in unique_visits:
        visit_df = df[df['Visit명'] == visit_name]

        # Retain '연구대상자ID' for merging
        visit_df_id = visit_df[['연구대상자ID']]

        # Drop unnecessary columns and rename remaining columns
        visit_df = visit_df.drop(columns=['과제번호', 'Visit명', '기관코드', '연구대상자ID', '연구대상자명', 'Random 번호', 'Random 배정일시', 'Random 배정자', 'Arm', 'Factors'], errors='ignore')
        if sheet_name != '1_인구통계학적 정보':
            visit_df = visit_df.drop(columns=['기관코드', '서면동의일', '일정시작일', '생년월일', '나이', '성별', '나이 단위', '나이단위', '연구대상자 등록일'], errors='ignore')

        # Rename columns
        renamed_columns = {col: f"{visit_name}_{sheet_name}_{col}" for col in visit_df.columns}
        visit_df = visit_df.rename(columns=renamed_columns)
        visit_df = pd.concat([visit_df_id, visit_df], axis=1)

        # Handle duplicate IDs
        if visit_df['연구대상자ID'].duplicated().any():
            duplicated_ids = visit_df[visit_df['연구대상자ID'].duplicated()]['연구대상자ID'].unique()
            for dup_id in duplicated_ids:
                dup_rows = visit_df[visit_df['연구대상자ID'] == dup_id]
                combined_row = dup_rows.groupby('연구대상자ID').agg(lambda x: x.dropna().unique().tolist()).reset_index()
                visit_df = visit_df[visit_df['연구대상자ID'] != dup_id]
                visit_df = pd.concat([visit_df, combined_row], axis=0)

        # Merge the visit-specific data with the combined DataFrame
        combined_df_backup = combined_df
        combined_df = pd.merge(combined_df, visit_df, on='연구대상자ID', how='left')
        if combined_df.shape[0] != num:
            ipdb.set_trace()

# Sort columns by the specified order
column_order = ['연구대상자ID', '기관코드']
visit_order = [ '투석 및 이식', 'Screening', 'Visit 1', 'Visit 2', 'Visit 3', 'Visit 4', 'Visit 5', 'Visit 6', 'Visit 7 / EoT', 'Post hoc', '연구 종결', 'Un-scheduled Visit', 'Adverse Event', 'Concomitant Medication']

# Append visit and sheet prefixes to the order
for visit in visit_order:
    column_order.extend([col for col in combined_df.columns if col.startswith(visit)])
# Reorder columns in the DataFrame
combined_df = combined_df[[col for col in column_order if col in combined_df.columns]]

# Replace all empty lists `[]` with NaN
combined_df = combined_df.applymap(lambda x: np.nan if isinstance(x, list) and not x else x)

# Remove columns with all values empty (NaN) and print the shape
combined_df = combined_df.dropna(axis=1, how='all')
print("Shape after removing empty columns:", df.shape)

# Process combined DataFrame
print("Initial shape of the combined dataframe:", combined_df.shape)
combined_df = combined_df.applymap(lambda x: np.nan if isinstance(x, list) and not x else x)
combined_df = combined_df.dropna(axis=1, how='all')

# Process columns containing lists
for col in combined_df.columns:
    if combined_df[col].apply(lambda x: isinstance(x, list)).any():
        max_list_length = combined_df[col].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
        if max_list_length == 1:
            combined_df[col] = combined_df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
        elif max_list_length > 1:
            combined_df[col] = combined_df[col].apply(lambda x: [x] if not isinstance(x, list) else x)

# Filter rows where specific column has value '예'
filter_column = '연구 종결_27_연구 종결_피험자가 본 임상시험의 모든 평가와 검사를 완료하였습니까?'
if filter_column in combined_df.columns:
    combined_df = combined_df[combined_df[filter_column] == '예']

# Keep only columns containing specified keywords
keywords = ['연구대상자ID', '기관코드', '투석 및 이식', 'Screening', 'Visit 1', 'Post hoc', 'EoT']
columns_to_keep = [col for col in combined_df.columns if any(keyword in col for keyword in keywords)]
combined_df = combined_df[columns_to_keep]

# Remove columns containing specified keywords
remove_keywords = ['동의일', '시작일', '날짜', '시행일', '생년월일', '나이단위', '등록일', '방문예정일', '방문일', 'eCRF', 'Version', '등록자', '생성일', '수정자', '수정일', '수집일', '측정일', '반복번호', '미수집', '질병 코드', '진단일', '처치일', '검사일', 'Specify', '채취일', '배정일', '작성하였습니까', '배정되었습니까', '교육진행여부', '연구대상자상태', '비고', '사유', '시행여부', '시행 여부', '구체적 기술', '선정제외기준', '실시일', '진단받으셨습니까', 'specify']
columns_to_remove = [col for col in combined_df.columns if any(keyword in col for keyword in remove_keywords)]
combined_df = combined_df.drop(columns=columns_to_remove, errors='ignore')

# Keep only columns containing specified keywords
keywords = ['연구대상자ID', '기관코드', '혈액투석여부', '복막투석여부', '신장이식여부', '비계획 투석 여부', 
            '인구통계학적 정보', '활력징후', '가족력유무', '피험자에게 1년 이내의 외과적 수술을 포함한 관련된 과거 및 현재 병력이 있습니까?', '흉부 방사선 평가', '심전도(12-lead ECG) 결과', '정상여부', '검사 결과', '정상 여부', '무작위 배정군', '설문지'] 
columns_to_keep = [col for col in combined_df.columns if any(keyword in col for keyword in keywords)]
combined_df = combined_df[columns_to_keep]

# Drop rows with missing values in specific columns
columns_to_check = [
    '투석 및 이식_22_투석 및 이식_혈액투석여부',
    '투석 및 이식_22_투석 및 이식_복막투석여부',
    '투석 및 이식_22_투석 및 이식_신장이식여부',
    '투석 및 이식_23_응급투석_비계획 투석 여부'
]
if all(col in combined_df.columns for col in columns_to_check):
    combined_df = combined_df.dropna(subset=columns_to_check, how='any')
else:
    missing_cols = [col for col in columns_to_check if col not in combined_df.columns]
    print(f"Columns not found in DataFrame: {missing_cols}")

# List of columns to check for non-empty values
columns_to_check = [
    'Screening_1_인구통계학적 정보_성별'
]
# Remove rows where any of these columns are NaN
if all(col in combined_df.columns for col in columns_to_check):
    combined_df = combined_df.dropna(subset=columns_to_check, how='any')
else:
    missing_cols = [col for col in columns_to_check if col not in combined_df.columns]


# Step #10: Convert lists into values
def process_lists1(value):
    if isinstance(value, list):
        if len(value) == 1:  # Single element list
            return value[0]
        elif len(value) == 0:  # Empty list
            return ''
    return value  # Return value unchanged if not a list
combined_df = combined_df.applymap(process_lists1)

# Step #10: Convert pseudo-lists into values
def process_lists2(value):
    # Try to convert strings that look like lists into actual lists
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        try:
            value = ast.literal_eval(value)  # Safely evaluate the string to convert it to a list
        except (ValueError, SyntaxError):
            pass  # Leave value as is if it can't be evaluated
    # Handle actual lists or pseudo-lists
    if isinstance(value, list):
        if len(value) == 1:  # Single element list
            return value[0]
        elif len(value) == 0:  # Empty list
            return ''
    return value  # Return value unchanged if not a list or pseudo-list
combined_df = combined_df.applymap(process_lists2)


# Save the preprocessed dataframe
print("Final df Shape:", combined_df.shape)

# Save the preprocessed dataframe
output_csv = f'p_{os.path.splitext(file_name)[0]}.csv'
output_excel = output_csv.replace('.csv', '.xlsx')

combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
combined_df.to_excel(output_excel, index=False, encoding='utf-8-sig')

print("Processing complete.")
print(f"CSV saved as: {output_csv}")
print(f"Excel saved as: {output_excel}")
