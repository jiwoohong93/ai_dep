'''
KoBBQ
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
'''

import torch
from tqdm.auto import tqdm
from transformers import pipeline

KOALPACA_MODEL = ['KoAlpaca-Polyglot-12.8B']
KOALPACA_MODEL_PATH = {'KoAlpaca-Polyglot-12.8B':'beomi/KoAlpaca-Polyglot-12.8B'}

def load_koalpaca(model_name='KoAlpaca-Polyglot-12.8B'):
    pipe = pipeline(
        'text-generation', 
        torch_dtype=torch.bfloat16, 
        model=KOALPACA_MODEL_PATH[model_name], 
        tokenizer=KOALPACA_MODEL_PATH[model_name],
        device_map="auto", 
        trust_remote_code=True)
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id
    pipe.tokenizer.padding_side = 'left'
    return pipe

def get_koalpaca_response(
    prompt,
    model_name,
    pipe,
    max_tokens,
    batch_size
):
    assert model_name in KOALPACA_MODEL
    result = []
    try:
        for idx, out in enumerate(tqdm(pipe(prompt, batch_size=batch_size, max_new_tokens=max_tokens), total=len(prompt))):
            raw = out[0]['generated_text']
            result.append(raw.split(prompt[idx])[-1])
    except Exception as e:
        print(e)
    return result