from openai import OpenAI
import json,re
import argparse
import os
def get_gpt4_score(text):
    pattern = r"the score is (\d+)"
    text = re.findall(pattern, text.lower())[0] if re.findall(pattern, text.lower())!=[] else text
    pattern = r"the score is:\n\n(\d+)"
    text = re.findall(pattern, text.lower())[0] if re.findall(pattern,text.lower())!=[] else text
    pattern = r"is(\d+)"
    text = text.replace('\n','').replace(':','')
    text = re.findall(pattern, text.lower())[0] if re.findall(pattern, text.lower())!=[] else text
    pattern = r"score (\d+)\*\*"
    text = re.findall(pattern, text.lower())[0] if re.findall(pattern, text.lower())!=[] else text
    pattern = r"is\s*(\d+)\."
    text = re.findall(pattern, text.lower())[0] if re.findall(pattern, text.lower())!=[] else text
    pattern = r"\*\*(\d+)\*\*"
    text = re.findall(pattern, text.lower())[0] if re.findall(pattern, text.lower())!=[] else text
    pattern = r"(\d+)the"
    text = re.findall(pattern, text.lower())[0] if re.findall(pattern, text.lower())!=[] else text
    score = int(round(float(text)))
    return score

parser = argparse.ArgumentParser()
parser.add_argument('--openapi_key', required=True)
parser.add_argument('--text', required=True)
args = parser.parse_args()

os.environ['OPENAI_API_KEY'] = args.openapi_key
client = OpenAI()

with open("prompt.json",'r') as f:
    file = json.load(f)
instruction,end = file[0]['application']['paradetox']
instruction,end

def chat_completions_with_backoff(**kwargs): # turbo, gpt4
    return client.chat.completions.create(**kwargs)       
def get_openai_response(text,instruction,end,modelname='gpt-4o-2024-05-13',temperature=0,):
    output = chat_completions_with_backoff(
                model = modelname, 
                temperature = temperature,
                messages = [
                    {'role': 'user', 'content': instruction + text + end},
                ],
            )
    res = output.choices[0].message.content
    return res
modelname = 'gpt-4o-2024-05-13'
res = get_openai_response(args.text,instruction,end,modelname)
print(get_gpt4_score(res))
