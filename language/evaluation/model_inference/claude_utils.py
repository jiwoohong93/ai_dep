'''
KoBBQ
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
'''

import os
import json
import time
import anthropic


CLAUDE_MODEL = ['claude-instant-1.2', 'claude-2.0']
CLAUDE_API_KEY = os.environ.get('CLAUDE')


def get_claude_response(
    prompt, 
    model_name, 
    max_tokens=128,
    max_try=10
):
    assert model_name in CLAUDE_MODEL
    
    n_try = 0
    while True:
        if n_try == max_try:
            raise Exception('Something Wrong')
        
        try:
            time.sleep(1)
            c = anthropic.Client(CLAUDE_API_KEY)
            resp = c.completion(
                prompt=f'{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}',
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model=model_name,
                max_tokens_to_sample=max_tokens
            )
            break
            
        except KeyboardInterrupt:
            raise Exception('KeyboardInterrupt')
        except Exception as e:
            print(e)
            print('Exception: Sleep for 5 sec')
            time.sleep(5)
            n_try += 1
            continue
            
    return resp['completion']
