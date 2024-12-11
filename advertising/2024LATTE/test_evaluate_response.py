
import pandas as pd
data_dir = './datas'

tox = pd.read_csv(f"{data_dir}/detox.csv")
tox_response = pd.read_csv("./gpt4o/Eval_detoxdf500_gpt-4o-2024-05-13.csv")
pattern = r"the score is (\d+)"
tox_response = tox_response.applymap(lambda x: re.findall(pattern, x.lower())[0] if re.findall(pattern, x.lower())!=[] else x)
pattern = r"the score is:\n\n(\d+)"
tox_response = tox_response.applymap(lambda x: re.findall(pattern, x.lower())[0] if re.findall(pattern, x.lower())!=[] else x)
pattern = r"is(\d+)"
tox_response = tox_response.applymap(lambda x: x.replace('\n','').replace(':',''))
tox_response = tox_response.applymap(lambda x: re.findall(pattern, x.lower())[0] if re.findall(pattern, x.lower())!=[] else x)
pattern = r"score (\d+)\*\*"
tox_response = tox_response.applymap(lambda x: re.findall(pattern, x.lower())[0] if re.findall(pattern, x.lower())!=[] else x)
pattern = r"is\s*(\d+)\."
tox_response = tox_response.applymap(lambda x: re.findall(pattern, x.lower())[0] if re.findall(pattern, x.lower())!=[] else x)
pattern = r"\*\*(\d+)\*\*"
tox_response = tox_response.applymap(lambda x: re.findall(pattern, x.lower())[0] if re.findall(pattern, x.lower())!=[] else x)
pattern = r"(\d+)the"
tox_response = tox_response.applymap(lambda x: re.findall(pattern, x.lower())[0] if re.findall(pattern, x.lower())!=[] else x)
tox_response = tox_response.applymap(lambda x:int(x))
tox['pred'] = tox_response['0']
tox0,tox1 = tox[tox['labels']==0],tox[tox['labels']==1]
print('Toxic / Safe Detection Accuracy')
print(sum(tox1['pred'] == tox1['labels']) / 250),print(sum(tox0['pred'] == tox0['labels']) / 250)
