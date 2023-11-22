import openai
import pdb
from tqdm import tqdm
import json
import math
import pandas as pd
import backoff
from tqdm import tqdm


N = 1
TEMP = 0
DATA_PATH = "fewshot_dataset/filtered_data_new_data.csv"




@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError, openai.error.Timeout))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

openai.api_key = "YOUR_OPEN_AI_API_KEY"







data = pd.read_csv(DATA_PATH, sep='_')
preds = []
with open('fewshot_dataset/filtered_data_novel_class_pred_12shot.csv', 'w') as f:
    f.write('text_label_pred\n')
    for row in tqdm(data.iterrows()):
        text = row[1]['text']
        # sentiment = row[1]['label'] if row[1]['label'] == 'negative' else 'positive'
        sentiment = row[1]['label']

#         prompt = '''
# 질병의 예방ᆞ치료에 효능이 있거나 신체기능강화에 도움을 주는 것으로 인식할 우려가 있는 광고 문구(질병예방치료_신체기능강화)와 그렇지 않은 광고 문구(negative)로 분류하시오.
# 아래는 분류 예시이고, 예시들은 답변하지 마시오.

# "유산균 김치 변비에 설사에 좋은 음식 피부 아토피 면역 숙변 배변": 질병예방치료_신체기능강화
# "심근경색과 뇌경색을 예방하는 천연 성분,": 질병예방치료_신체기능강화
# "좁쌀여드름연고 트러블은 3일정도 쓰니깐 좋아지구요 땀띠 * 트러블 진정에 탁월하며": 질병예방치료_신체기능강화
# "제대로 키운 열매를 순수하고 건강한 자연의 맛 그대로 건강한 먹거리 문화 조성에 정진하겠습니다.": negative
# "피곤하고 지칠때 한잔의 리얼 비타민 상쾌한 기분전환": negative
# "유기농 채소와 곡물, 과일을 재료로 사용합니다.": negative

# 아래 문장은 질병예방치료_신체기능강화 과 negative 둘 중 어느 것인지 분류하시오. 분류 예시와 동일한 template 을 사용하여 대답하시오.

# '''

        prompt = '''
질병의 예방ᆞ치료에 효능이 있거나 신체기능강화에 도움을 주는 것으로 인식할 우려가 있는 광고 문구(positive)와 그렇지 않은 광고 문구(negative)로 분류하시오.
아래는 분류 예시이고, 예시들은 답변하지 마시오.

"유산균 김치 변비에 설사에 좋은 음식 피부 아토피 면역 숙변 배변": positive
"심근경색과 뇌경색을 예방하는 천연 성분,": positive
"좁쌀여드름연고 트러블은 3일정도 쓰니깐 좋아지구요 땀띠 * 트러블 진정에 탁월하며": positive
"독일산 맥주효모 프랑스산 남성 여성 탈모에좋은 머리카락 영양제 새치 흰머리": positive
"초음파 가정용 휴대용 아기 천식 비염 폐렴 호흡기치료기 네블라이저": positive
"목병의 치료에 탁월한 효과가 있어 편도선이 붓고 아픈 것을 잘 낫게 하고 인후염에 좋습니다.": positive
"제대로 키운 열매를 순수하고 건강한 자연의 맛 그대로 건강한 먹거리 문화 조성에 정진하겠습니다.": negative
"피곤하고 지칠때 한잔의 리얼 비타민 상쾌한 기분전환": negative
"유기농 채소와 곡물, 과일을 재료로 사용합니다.": negative
"아미노파워 최적의 아미노산 믹스 바디밸런스를 위한 최적의 아미노산믹스(BCAA,아르기닌 등)": negative
"특허성분과 임상으로 입증된 숙취해소효과 특허성분과 임상실험으로 입증된 확실한 숙취해소 효과를 느껴보세요!": negative
"흡수가 빠른 분리유청단백(Whey Protein) 단백질은 종류에 따라 우리 몸에서 소화/흡수되는 속도가 다릅니다. 운동 후, 자극된 근육에 빠르게 아미노산을 공급할 수 있는 흡수빠른 분리유청단백질을 섭취하는 것이 중요합니다.": negative

아래 문장은 positive 과 negative 둘 중 어느 것인지 분류하시오. 분류 예시와 동일한 template 을 사용하여 대답하시오.

'''


        prompt = prompt + '"' + text + '"' + ": "
        completion = completion_with_backoff(
            model="gpt-4", 
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=1,
            temperature=TEMP,
            n=N
        )
        pred = completion['choices'][0]['message']['content']
        print(pred)
        if 'negative' in pred:
            pred = 'negative'
        elif 'positive' in pred:
            pred = '"질병예방치료_신체기능강화"'
        newline = text + '_' + row[1]['label'] + '_' + pred
        f.write(newline + '\n')
