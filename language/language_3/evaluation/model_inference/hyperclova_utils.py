'''
KoBBQ
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
'''

import os
import time
import json
import requests


HYPERCLOVA_MODEL = {'clova-x': os.environ.get('CLOVA_URL')}
HEADERS = {
    'Content-Type': 'application/json; charset=utf-8',
    'Authorization': f'Bearer {os.environ.get("CLOVA")}'    
}


def get_hyperclova_response(
    text,
    model_name,
    greedy,
    temperature=None,
    top_p=None,
    max_tokens=128,
    repeat_penalty=3,
    max_try=10
):
    assert model_name in HYPERCLOVA_MODEL
    
    data = {
        'text_batch': text,
        'greedy': greedy,
        'max_tokens': max_tokens,
        'recompute': False,
        'repeat_penalty': repeat_penalty
    }
    if not greedy:
        data['temperature'] = temperature
        data['top_p']: top_p

    n_try = 0
    while True:
        if n_try > max_try:
            raise Exception('Something Wrong')

        try:
            response = requests.post(f'{HYPERCLOVA_MODEL[model_name]}',
                                     headers=HEADERS,
                                     data=json.dumps(data),
                                     timeout=60)
            
            if response.status_code != 200:
                print(f'Error from internal API: {response.status_code}')
                time.sleep(5)
                n_try += 1
                continue
            
            outputs = response.json()['results']
            results = [output['text'].strip().replace(prompt, '') for output, prompt in zip(outputs, data['text_batch'])]
            break

        except KeyboardInterrupt:
            raise Exception('KeyboardInterrupt')
        except Exception as e:
            print(e)
            print('Exception: Sleep for 5 sec')
            time.sleep(5)
            n_try += 1
            continue
            
    return results
