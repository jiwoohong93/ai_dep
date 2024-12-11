import pandas as pd
import re
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
data_dir = './datas'

bbq = pd.read_csv(f"{data_dir}/bbq.csv") # (A)
bbq_response = pd.read_csv("./gpt4o/Partiality_gpt-4o-2024-05-13.csv")
pattern = r'\(*([ABC]+)\)*'
bbq_response = bbq_response.applymap(lambda x: re.findall(pattern, x)[0])
bbq_response = bbq_response.applymap(lambda x: 1 if 'A' in x else 0)
print("Partiality : ", round(sum(bbq_response['0']) / 1100,3))
