import numpy as np
import openai
import os
import sys
import time
import asyncio
import aiohttp


# Uniform w/o replacement sampler
class UWORetriever:
    def __init__(self, dataset, test_dataset, template, seed=43):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.seed = seed
        self.prompt_template = template
        self.all_ensemble_idx_list = None

    def retrieve(self, ice_num, ensemble_num, test_ds):
        np.random.seed(self.seed)
        num_idx = len(self.dataset)
        rtr_idx_list = []
        for _ in range(test_ds):
            idx_list = np.random.choice(
                num_idx, ice_num * ensemble_num, replace=False
            ).tolist()  # Sample N*M ICE from the dataset
            rtr_idx_list.append(idx_list)
        all_ensemble_idx_list = []

        for i in range(ensemble_num):
            idx_list_temp = []
            for j in range(len(rtr_idx_list)):
                idx_list_temp.append(rtr_idx_list[j][i * ice_num : (i + 1) * ice_num])
            all_ensemble_idx_list.append(idx_list_temp)

        self.all_ensemble_idx_list = all_ensemble_idx_list
        return all_ensemble_idx_list

    def prompt_generate(self, query_idx, ice_num, ensemble_num, output_field=None):
        ensemble_idx = [e[1] for e in self.all_ensemble_idx_list]
        test = self.test_dataset[query_idx]
        query = self.prompt_template.generate_item(test, output_field=output_field)
        prompt_list = []

        for i in range(ensemble_num):
            prompt_temp = ""
            # If ice_num is greater than 0, generate in-context examples
            if ice_num > 0:
                for j in range(ice_num):
                    idx = ensemble_idx[i][j]  # ith ensemble jth ICE
                    output = self.prompt_template.generate_item(self.dataset[idx])
                    if j == 0:
                        prompt_temp += output
                    else:
                        prompt_temp += "\n" + output

                prompt_temp += "\n"  # Only add newline after ICE if ice_num > 0

            # Add the query after ICE or by itself if zero-shot
            prompt_temp += query
            prompt_list.append(prompt_temp)

        return prompt_list


async def complete(
    prompt,
    l,
    temp,
    model_name,
    num_log_probs=None,
    echo=False,
    n=None,
    instruction=None,
):
    # Set your OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # GPT-3 API URL
    if "turbo" in model_name or "4o" in model_name:
        api_url = "https://api.openai.com/v1/chat/completions"
        message = [{"role": "user", "content": prompt}]
        if instruction:
            message.append(
                {
                    "role": "system",
                    "content": instruction,
                }
            )

        data = {
            "model": model_name,
            "messages": message,
            "max_tokens": l,
            "temperature": temp,
            "n": n,
            "stop": ["\n"],
        }
    else:
        if instruction:
            prompt = instruction + prompt
        api_url = "https://api.openai.com/v1/completions"
        data = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": l,
            "temperature": temp,
            "logprobs": num_log_probs,
            "n": n,
            "echo": echo,
            "stop": "\n",
        }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    response = None
    received = False

    async with aiohttp.ClientSession() as session:
        while not received:
            try:
                async with session.post(api_url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()

                        if "turbo" in model_name or "4o" in model_name:
                            text = result["choices"][0]["message"]["content"]
                        else:
                            text = result["choices"][0]["text"]

                        received = True if text.strip() != "" else False
                    else:
                        print(f"Error {resp.status}: {await resp.text()}")
                        await asyncio.sleep(1)
            except Exception as error:
                print("API error:", error)
                await asyncio.sleep(1)

    return text


'''
def complete(prompt, l, temp, model_name, num_log_probs=None, echo=False, n=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    while not received:
        try:
            if "turbo" in model_name:
                response = openai.ChatCompletion.create(
                    model=model_name,  # Use "gpt-3.5-turbo" or your chosen model
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a doctor. Please answer medical questions based on the patient's description.",
                        },  # Optional system message
                        {
                            "role": "user",
                            "content": prompt,
                        },  # The prompt provided by the user
                    ],
                    max_tokens=l,
                    temperature=temp,
                    logprobs=num_log_probs,
                    n=n,
                    stop=["\n"],
                )
                text = response["choices"][0]["message"]["content"]
            else:
                response = openai.Completion.create(
                    engine=model_name,
                    prompt=prompt,
                    max_tokens=l,
                    temperature=temp,
                    logprobs=num_log_probs,
                    echo=echo,
                    stop="\n",
                    n=n,
                )
                text = response["choices"][0]["text"]
            received = True if text.strip() != "" else False
        except:
            error = sys.exc_info()[0]
            """
            if (
                error == openai.Invalid
            ):  # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            """
            print("API error:", error)
            time.sleep(1)

    return text
'''
