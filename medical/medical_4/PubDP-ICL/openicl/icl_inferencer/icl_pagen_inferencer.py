"""Direct Generation Inferencer"""

import json
import torch
from openicl import PromptTemplate
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from openicl.utils.api_service import *
from openicl.utils.icl_common_utils import get_dataloader, get_generation_prompt_list_from_retriever_indices
from openicl.utils.logging import get_logger
from openicl.utils.sample import sample, norm_logits, top_k_top_p_filter, max_fn, autoregressive_sampling
from typing import List, Union, Optional
from tqdm import tqdm
from transformers import PretrainedConfig
from accelerate import Accelerator
import time


logger = get_logger(__name__)


class PAGenInferencer(BaseInferencer):
    """Generation In-context Learning Inferencer Class
        In-context Learning Inferencer for Directly Generation.

    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class. 
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM. 
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file. 
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file. 
        api_name (:obj:`str`, optional): Name of API service. 
        call_api (:obj:`bool`): If ``True``, an API for LM models will be used, determined by :obj:`api_name`.   
        gen_field_replace_token (:obj:`str`, optional): Used to replace the generation field token when generating prompts.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the :obj:`model.generate()` method. 
    """

    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 gen_field_replace_token: Optional[str] = '',
                 generation_kwargs={"max_new_tokens": 100},
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,
                 args: Optional[object] = None,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs
        # extract values from kwargs
        self.args = args

    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None, force_words=None, skip_num=0) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = GenInferencerOutputHandler(num, self.accelerator)
        index = 0

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()
        # print("ice_idx_list", ice_idx_list)

        # 3. Generate prompts for testing input
        prompt_list_ensmble = []
        for idx_list in ice_idx_list:
            prompt_list = get_generation_prompt_list_from_retriever_indices(idx_list, retriever, self.tokenizer,
                                                                            self.gen_field_replace_token,
                                                                            max_model_token_num=self.max_model_token_num,
                                                                            ice_template=ice_template,
                                                                            prompt_template=prompt_template)
            prompt_list_ensmble.append(prompt_list)
        # print(len(prompt_list_ensmble), len(prompt_list_ensmble[0]))

        prompt_list_rearranged = []
        for j in range(len(prompt_list_ensmble[0])):
            for i in range(len(prompt_list_ensmble)):
                prompt_list_rearranged.append(prompt_list_ensmble[i][j])

        # print(prompt_list_rearranged, len(prompt_list_rearranged))
        prompt_list = prompt_list_rearranged
        output_handler.save_orgin_prompts(prompt_list)

        # 4. Wrap prompts with Dataloader
        dataloader = get_dataloader(prompt_list, self.batch_size)

        # 5. Inference for prompts in each batch
        logger.info("Starting inference process...")
        print(len(dataloader))
        for entry in tqdm(dataloader, disable=not self.is_main_process):
            if index < skip_num:
                output_handler.save_prediction_and_output("", "", index)
                index = index + 1

                continue
            # print(entry)
            # 5-1. Inference with local model
            if not self.call_api:
                with torch.no_grad():
                    for ind, en in enumerate(entry):
                        prompt_len = 10000
                        if self.args.data_name == "ocr":
                            while True:
                                tokenized_data = self.tokenizer.batch_encode_plus([en], padding=True, return_tensors='pt').to(
                                    self.device)
                                prompt_len = int(
                                    tokenized_data.attention_mask.shape[1])
                                if prompt_len < 2000:
                                    break
                                else:
                                    print("prompt_len", prompt_len)
                                    prev_prompt = en[:32]
                                    first_comma_index = en.find(",")
                                    second_comma_index = en.find(
                                        ",", first_comma_index+1)
                                    first_item = en[first_comma_index: second_comma_index]
                                    other_text = en[second_comma_index:]
                                    entry[ind] = prev_prompt + other_text
                                    en = entry[ind]
                                    # print("entry[ind]", len(entry[ind]))
                    tokenized_data = self.tokenizer.batch_encode_plus(entry, padding=True, return_tensors='pt').to(
                        self.device)
                    prompt_len = int(tokenized_data.attention_mask.shape[1])
                    # print("prompt_len", prompt_len)
                    # print(tokenized_data)
                    if 't5' in self.model_name:
                        prompt_len = 0
                    if force_words is not None:
                        force_words_ids = [
                            self.tokenizer(force_words).input_ids,
                        ]
                        outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      force_words_ids=force_words_ids,
                                                      num_beams=10,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                    else:
                        outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=[
                                                          self.tokenizer.eos_token_id, 434],
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)

                    print("outputs", len(outputs))

                    outputs = outputs.tolist()
                    complete_output = self.tokenizer.batch_decode(
                        outputs[:], skip_special_tokens=True)
                    generated = self.tokenizer.batch_decode([output[prompt_len:] for output in outputs],
                                                            skip_special_tokens=True)

            # 5-2. Inference with remote API
            else:
                complete_output, generated = api_get_tokens(
                    self.api_name, entry)

            # print(generated)

            # 5-3. Save current output
            for prediction, output in zip(generated, complete_output):
                output_handler.save_prediction_and_output(
                    prediction, output, index)
                output_handler.write_now(
                    output_json_filepath, output_json_filename)
                index = index + 1

            # 6. Output
        output_handler.subprocess_write_to_json(
            output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(
            output_json_filepath, output_json_filename)
        output_handler.write_to_json(
            output_json_filepath, output_json_filename)
        return [sample['prediction'] for sample in output_handler.results_dict.values()]

    def embedding(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None, input_json_filename: Optional[str] = None, force_words=None) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = GenInferencerOutputHandler(num, self.accelerator)
        index = 0

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()
        # print("ice_idx_list", ice_idx_list)

        # 3. Generate prompts for testing input
        prompt_list_ensmble = []
        for idx_list in ice_idx_list:
            prompt_list = get_generation_prompt_list_from_retriever_indices(idx_list, retriever, self.tokenizer,
                                                                            self.gen_field_replace_token,
                                                                            max_model_token_num=self.max_model_token_num,
                                                                            ice_template=ice_template,
                                                                            prompt_template=prompt_template)
            prompt_list_ensmble.append(prompt_list)
        # print(len(prompt_list_ensmble), len(prompt_list_ensmble[0]))

        prompt_list_rearranged = []
        for j in range(len(prompt_list_ensmble[0])):
            for i in range(len(prompt_list_ensmble)):
                prompt_list_rearranged.append(prompt_list_ensmble[i][j])

        print(prompt_list_rearranged, len(prompt_list_rearranged))
        prompt_list = prompt_list_rearranged
        output_handler.save_orgin_prompts(prompt_list)

        # 4. Wrap prompts with Dataloader
        # dataloader = get_dataloader(prompt_list, self.batch_size)

        # 5. Inference for prompts in each batch
        logger.info("Starting inference process...")

        # read the output file
        with open('icl_inference_output/'+input_json_filename+'.json', 'r') as f:
            predictions = json.load(f)
        print("predictions", len(predictions))
        # extract tht output from the predictions
        output = []
        origin_prompt = []
        for i in range(len(predictions)):
            # predictions[str(i)]['output'].find("Summarize the above dialogue:")
            # print([predictions[str(i)]['output'][predictions[str(i)]['output'].rfind("Dialogue:"):]])
            # find the last "Summarize the above dialogue" in output:
            output.append([predictions[str(i)]['output']
                          [predictions[str(i)]['output'].rfind("Dialogue:"):]])
            origin_prompt.append([predictions[str(i)]['origin_prompt'][predictions[str(
                i)]['origin_prompt'].rfind("Dialogue:"):]])
        print(retriever.ensemble, "length of predictions", len(predictions))
        prediction_length = len(predictions) // retriever.ensemble

        all_embedding = []
        for index in tqdm(range(len(predictions))):
            curr_output = output[index]
            embedding = api_get_embedding(self.api_name, curr_output)
            all_embedding.append(embedding)
        print("all_embedding", len(all_embedding))

        # Create dataframe to store the embedding
        import pandas as pd
        df = pd.DataFrame(all_embedding)
        df.to_csv('embedding/'+output_json_filename +
                  'embedding.csv', index=False)

        # stop the program
        print("stop the program")
        import sys
        sys.exit(0)

        return all_embedding


def DP_select(next_token_list):
    # differentially private select a token from the next_token_list
    dict = {}
    for token in next_token_list:
        if token in dict:
            dict[token] += 1
        else:
            dict[token] = 1
    print("dict", dict)
    for key in dict:
        dict[key] = dict[key] + np.random.laplace(0, 1, 1)[0]
    print("dict", dict)
    return max(dict, key=dict.get)
