import json
import requests
import os
import openai
import time
import numpy as np

OPENICL_API_NAME_LIST = ['opt-175b', 'gpt3', 'gpt35']
OPENICL_API_PARAMETER_DICT = {
    'opt-175b': ['URL', 'headers'],
    'gpt3': ['engine', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'sleep_time'],
    'gpt35': ['engine', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'sleep_time']

}
OPENICL_API_REQUEST_CONFIG = {
    'opt-175b': {
        'URL': "",  # http://xxx/completions or http://xxx/generate
        'headers': {
            "Content-Type": "application/json; charset=UTF-8"
        }
    },
    'gpt3': {
        'engine': "text-davinci-003",
        'temperature': 0,
        'max_tokens': 256,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'sleep_time': 3,
        'logprobs': 0,
        'best_of': 1
    },
    'gpt35': {
        'engine': "text-davinci-003",
        'temperature': 0,
        'max_tokens': 256,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'sleep_time': 1    }
}
PROXIES = {"https": "", "http": ""}


def is_api_available(api_name):
    if api_name is None:
        return False
    return True if api_name in OPENICL_API_NAME_LIST else False


def update_openicl_api_request_config(api_name, **kwargs):
    if api_name is None or not is_api_available(api_name):
        return

    parameter_list = OPENICL_API_PARAMETER_DICT[api_name]
    for parameter in parameter_list:
        if parameter in kwargs.keys():
            print(f"Update {api_name} API parameter: {parameter} = {kwargs[parameter]}")
            OPENICL_API_REQUEST_CONFIG[api_name][parameter] = kwargs[parameter]


def api_get_ppl(api_name, input_texts):
    if api_name == 'opt-175b':
        pyload = {"prompt": input_texts, "max_tokens": 0, "echo": True}
        response = json.loads(
            requests.post(OPENICL_API_REQUEST_CONFIG[api_name]['URL'], data=json.dumps(pyload),
                          headers=OPENICL_API_REQUEST_CONFIG[api_name]['headers'], proxies=PROXIES).text)
        lens = np.array([len(r['logprobs']['tokens']) for r in response['choices']])
        ce_loss = np.array([-sum(r['logprobs']['token_logprobs']) for r in response['choices']])
        return ce_loss / lens

    if api_name == 'gpt3':
        raise NotImplementedError("GPT-3 API doesn't support PPL calculation")


def api_get_tokens(api_name, input_texts):

    length_list = [len(text) for text in input_texts]

    if api_name == 'opt-175b':
        pyload = {"prompt": input_texts, "max_tokens": 100, "echo": True}
        response = json.loads(
            requests.post(OPENICL_API_REQUEST_CONFIG[api_name]['URL'], data=json.dumps(pyload),
                          headers=OPENICL_API_REQUEST_CONFIG[api_name]['headers'], proxies=PROXIES).text)
        return [r['text'] for r in response['choices']], [r['text'][length:] for r, length in
                                                          zip(response['choices'], length_list)]

    if api_name == 'gpt3':
        while True:
            try:
                print(OPENICL_API_REQUEST_CONFIG['gpt3'], len(input_texts[0])) 
                openai.api_key = os.getenv("OPENAI_API_KEY")
                response = openai.Completion.create(
                    engine=OPENICL_API_REQUEST_CONFIG['gpt3']['engine'],
                    prompt=input_texts,
                    temperature=OPENICL_API_REQUEST_CONFIG['gpt3']['temperature'],
                    max_tokens=OPENICL_API_REQUEST_CONFIG['gpt3']['max_tokens'],
                    top_p=OPENICL_API_REQUEST_CONFIG['gpt3']['top_p'],
                    frequency_penalty=OPENICL_API_REQUEST_CONFIG['gpt3']['frequency_penalty'],
                    presence_penalty=OPENICL_API_REQUEST_CONFIG['gpt3']['presence_penalty'],
                    best_of=OPENICL_API_REQUEST_CONFIG['gpt3']['best_of']
                )
                return [(input + r['text']) for r, input in zip(response['choices'], input_texts)], [r['text'] for r in
                                                                                                    response['choices']]
            except:
                time.sleep(OPENICL_API_REQUEST_CONFIG['gpt3']['sleep_time'])
                continue

    if api_name == 'gpt35':
        print(OPENICL_API_REQUEST_CONFIG['gpt35'], len(input_texts[0])) 
        while True:
            try:
                print("input_texts", input_texts)
                print(OPENICL_API_REQUEST_CONFIG['gpt35']['engine'])
                openai.api_key = os.getenv("OPENAI_API_KEY")
                response = openai.ChatCompletion.create(
                    model=OPENICL_API_REQUEST_CONFIG['gpt35']['engine'],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_texts[0]},
                    ],
                    temperature=OPENICL_API_REQUEST_CONFIG['gpt35']['temperature'],
                    max_tokens=OPENICL_API_REQUEST_CONFIG['gpt35']['max_tokens'],
                    top_p=OPENICL_API_REQUEST_CONFIG['gpt35']['top_p'],
                    frequency_penalty=OPENICL_API_REQUEST_CONFIG['gpt35']['frequency_penalty'],
                    presence_penalty=OPENICL_API_REQUEST_CONFIG['gpt35']['presence_penalty']        )
                print("response", response['choices'][0]['message']['content'])
                return [input_texts[0]+response['choices'][0]['message']['content']], [response['choices'][0]['message']['content']]
            except:
                time.sleep(OPENICL_API_REQUEST_CONFIG['gpt35']['sleep_time'])
                continue

def api_get_embedding(api_name, input_texts):
    if api_name == 'gpt3':
        openai.api_key = os.getenv("OPENAI_API_KEY")
        input_texts = input_texts[0].replace("\n", " ")
        embedding = openai.Embedding.create(input=input_texts, model="text-embedding-ada-002")["data"][0]["embedding"]
        # embedding = openai.Embedding.create(input=input_texts, model="text-embedding-ada-002")["data"][0]["embedding"]

        time.sleep(OPENICL_API_REQUEST_CONFIG['gpt3']['sleep_time'])
        return embedding


def api_get_one_token(api_name, input_texts, args):
    print("input_texts", len(input_texts))
    max_token_length = len(input_texts)//args.ensemble
    print("max_token_length", max_token_length)
    original_input_texts = input_texts
    length_list = [len(text) for text in input_texts]
    output_texts = ""

    if api_name == 'gpt3':
        openai.api_key = os.getenv("OPENAI_API_KEY")
        for index in range(max_token_length):
            curr_input_texts = input_texts[index*args.ensemble:(index+1)*args.ensemble]
            #print("curr_input_texts", curr_input_texts, len(curr_input_texts), len(input_texts))
            curr_input_texts = [curr_input_text + output_texts for curr_input_text in curr_input_texts]
            assert len(curr_input_texts) == args.ensemble
            #print("curr_input_texts", curr_input_texts, len(curr_input_texts[0]))
            response = openai.Completion.create(
                engine=OPENICL_API_REQUEST_CONFIG['gpt3']['engine'],
                prompt=curr_input_texts,
                temperature=OPENICL_API_REQUEST_CONFIG['gpt3']['temperature'],
                max_tokens=OPENICL_API_REQUEST_CONFIG['gpt3']['max_tokens'],
                top_p=OPENICL_API_REQUEST_CONFIG['gpt3']['top_p'],
                frequency_penalty=OPENICL_API_REQUEST_CONFIG['gpt3']['frequency_penalty'],
                presence_penalty=OPENICL_API_REQUEST_CONFIG['gpt3']['presence_penalty'],
                logprobs = 5
            )
            #print("response", response)
            big_dict = {}
            print(response['choices'][0]['logprobs']['top_logprobs'])
            num = 0
            for i in range(args.ensemble):
                if response['choices'][i]['logprobs']['top_logprobs'] == []:
                    num += 1
            if num != 0:
                break
                
            for i in range(args.ensemble):
                for k, v in response['choices'][i]['logprobs']['top_logprobs'][0].items():
                    ## log probs to probs
                    v = np.exp(v)
                    if k in big_dict.keys():
                        big_dict[k] += v
                    else:
                        big_dict[k] = v
            # print(big_dict)
            # find the most common token
            next_token = max(big_dict, key=big_dict.get)
            output_texts += next_token 
            print("next_token", output_texts, next_token)
            # if next_token is the end of the sentence, break
            if next_token == "<|endoftext|>":
                break
        # concatenate all output_tokens to one string
        # output_texts = ["".join(output_texts)]
        # delete all '\n' in the beginning of output_texts
        # output_texts = [output_text.replace("\n", " ") for output_text in output_texts]
        # add one '\n' to the beginning of the output_texts
        # output_texts = ["\n" + output_text for output_text in output_texts]
        time.sleep(OPENICL_API_REQUEST_CONFIG['gpt3']['sleep_time'])
        return [original_input_texts[0] + output_texts[0]], [output_texts]
    
    if api_name == 'gpt35':
        print("input_texts", input_texts)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant and going to summarize some dialogues."},
                {"role": "user", "content": input_texts},
            ],
            temperature=OPENICL_API_REQUEST_CONFIG['gpt35']['temperature'],
            max_tokens=OPENICL_API_REQUEST_CONFIG['gpt35']['max_tokens'],
            top_p=OPENICL_API_REQUEST_CONFIG['gpt35']['top_p'],
            frequency_penalty=OPENICL_API_REQUEST_CONFIG['gpt35']['frequency_penalty'],
            presence_penalty=OPENICL_API_REQUEST_CONFIG['gpt35']['presence_penalty'],
            logprobs=5
        )
        print("response", response)
        time.sleep(OPENICL_API_REQUEST_CONFIG['gpt35']['sleep_time'])
        return [(input + r['text']) for r, input in zip(response['choices'], input_texts)], [r['text'] for r in
                                                                                             response['choices']]