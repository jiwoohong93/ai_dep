'''
command:
python ASR_INTapt.py --do_model_download --eval_mode intapt --eval_metric wer

# 처음 코드 실행 시에는 --do_model_download 옵션을 사용하여 hubert 및 prompt generator 모델을 다운로드 받아야 합니다.
# 이후에는 --do_model_download 옵션을 사용하지 않아도 됩니다.
# --eval_mode 옵션을 사용하여 intapt 또는 base를 선택할 수 있습니다.
# --eval_metric 옵션을 사용하여 wer 또는 cer을 선택할 수 있습니다.
# 현재 코드는 coraal 데이터셋을 사용하도록 되어 있습니다.
# hf_cache_dir 은 huggingface cache directory를 지정하는 것으로, 해당 디렉토리에 모델을 다운로드 받습니다.
# 현재는 INTapt/hf_cache 로 설정되어 있습니다.
# 결과는 각 화자별로 wer/cer을 계산하고, 모든 화자에 대한 평균 wer/cer을 계산합니다.
# base 모델은 prompt를 사용하지 않고, intapt 모델은 prompt를 사용합니다.
# prompt를 사용하는 경우, prompt generator 모델을 사용하여 prompt를 생성합니다.
# prompt generator 모델은 INTapt-HuBERT-large-coraal-prompt-generator 모델을 사용합니다.
# batch_size는 8로 설정되어 있습니다. --batch_size 옵션을 사용하여 변경할 수 있습니다.
# prompt_length는 prompt의 길이로, 현재는 40으로 설정되어 있습니다. 학습된 모델이 40을 사용하기 때문에 변경하지 않는 것을 권장합니다.

'''

import os
import sys
import argparse

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(parent_dir)

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import model_utils

from transformers import Wav2Vec2Processor, HubertForCTC
from huggingface_hub import snapshot_download, hf_hub_download
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from evaluate import load
from data_utils import DataCollatorCTCWithPaddingCoraal
import utils
import glob

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

def load_model(args, ):
	model_path = glob.glob(os.path.join(current_dir, args.hf_cache_dir, "models--facebook--hubert-large-ls960-ft", "snapshots") +  "/*")[0]
	prompt_generator_path = glob.glob(os.path.join(current_dir, args.hf_cache_dir, "models--esyoon--INTapt-HuBERT-large-coraal-prompt-generator", "snapshots") +  "/*")[0]
	processor = Wav2Vec2Processor.from_pretrained(model_path)
	model = HubertForCTC.from_pretrained(model_path)

	prompt_generator = model_utils.PromptGenerator(args, model.config)

	prompt_generator.load_state_dict(torch.load(os.path.join(prompt_generator_path, "prompt_generator.pt")))

	model.to(args.device)
	prompt_generator.to(args.device)

	return processor, model, prompt_generator


def cal_logits_w_prompt(args, model, batch, prompt):
	batch['feature'] = model.hubert.feature_extractor(batch['input_values'])
	batch['feature'] = model.hubert.feature_projection(batch['feature'].transpose(1,2))
	model_input = torch.cat([prompt, batch['feature']], dim=1)
	orig_hidden_states = model(batch['input_values'], return_dict='pt', output_hidden_states=True)['hidden_states']
	pred_hidden_states_temp = model.hubert.encoder(model_input, return_dict='pt', output_hidden_states=True)
	last_hidden_state = pred_hidden_states_temp[0]
	pred_hidden_states = pred_hidden_states_temp[1]
	orig_hidden_state = orig_hidden_states[3]
	prompted_hidden_state = pred_hidden_states[3][:,args.prompt_length:,:]

	logits = model.dropout(last_hidden_state)
	logits = model.lm_head(logits[:,args.prompt_length:,:])

	return batch, orig_hidden_state, prompted_hidden_state, logits

def inference(args, model, prompt_generator, processor, metric, test_dataloader):
	model.eval()
	if args.eval_mode == 'intapt':
		prompt_generator.eval()

	total_wer = 0.
	steps = torch.tensor(len(test_dataloader)).to(args.device)

	for _, batch in enumerate(tqdm(test_dataloader)):
		batch = utils.dict_to_device(batch, args.device)
		if args.eval_mode == 'intapt':
			orig_pred = model(input_values=batch['input_values'], labels=batch['labels'], output_hidden_states=True)
			prompt = prompt_generator(orig_pred.hidden_states[3])
					
			_, _, _, prompt_logits = cal_logits_w_prompt(args, model, batch, prompt)

			wer = utils.compute_metrics(prompt_logits, batch['labels'], processor, metric)
		
		elif args.eval_mode == 'base':
			orig_pred = model(input_values=batch['input_values'], labels=batch['labels'], output_hidden_states=True)
			wer = utils.compute_metrics(orig_pred.logits, batch['labels'], processor, metric)

	total_wer += torch.tensor(wer['wer']).to(args.device)

	return total_wer/steps


def get_args():
	parser = argparse.ArgumentParser(description='CORAAL ASR test codes!')

	parser.add_argument("--hf_cache_dir", type=str, default="hf_cache")
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--dataset_name", type=str, default="coraal")
	parser.add_argument("--do_model_download", action="store_true")
	parser.add_argument("--eval_metric", type=str, default="wer")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--prompt_length", type=int, default=40)
	parser.add_argument("--eval_mode", type=str, default="intapt")

	return parser.parse_args()

def main():

	args = get_args()

	if args.eval_mode not in ['intapt', 'base']:
		print("Invalid eval mode. Choose from 'intapt' or 'base'...")
		quit()
	
	if not os.path.exists(args.hf_cache_dir):
		os.makedirs(args.hf_cache_dir, exist_ok=True)

	
	if not args.do_model_download:
		prompt_download_path = os.path.join(current_dir, args.hf_cache_dir, "models--esyoon--INTapt-HuBERT-large-coraal-prompt-generator")
		if not os.path.exists(prompt_download_path):
			print("Need to download the prompt generator model...")
			quit()

	if args.do_model_download:
		prompt_download_path = os.path.join(current_dir, args.hf_cache_dir, "models--esyoon--INTapt-HuBERT-large-coraal-prompt-generator")
		if os.path.exists(prompt_download_path):
			print("Prompt generator model already exists in the cache directory...")
		snapshot_download("esyoon/INTapt-HuBERT-large-coraal-prompt-generator", repo_type="model", cache_dir=args.hf_cache_dir)


	# load model
	processor, model, prompt_generator = load_model(args)
	
	# load metric
	if args.eval_metric == 'wer':
		metric = load("wer")
	elif args.eval_metric == 'cer':
		metric = load("cer")

	# load dataset
	data_collator = DataCollatorCTCWithPaddingCoraal(processor=processor, padding=True)

	# If there is no datasset in the cache directory, download the dataset
	test_dataset = load_dataset('esyoon/coraal_clean_test', cache_dir=args.hf_cache_dir)
	test_dataset = test_dataset['train']

	# set the all speakers to test
	test_speakers = ['ATL', 'DCA', 'DCB', 'LES', 'PRV', 'ROC'] 
	test_result_list = []

	for test_speaker in test_speakers:
		test_dataset_ = test_dataset.filter(lambda x: x['accent'] == test_speaker)
		test_dataloader = DataLoader(test_dataset_, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator, pin_memory=True)


		print("start testing for", test_speaker, "...")
		if args.eval_mode == 'intapt':
			result = inference(args, model, prompt_generator, processor, metric, test_dataloader)
		elif args.eval_mode == 'base':
			result = inference(args, model, None, processor, metric, test_dataloader)
		print(args.eval_metric, "for", test_speaker, ": ", result.item())
		test_result_list.append(result.item())

	
	print("test avg performance: {:.4f} ".format(sum(test_result_list)/len(test_result_list)))
	print("max - min performance: {:.4f}".format(max(test_result_list)-min(test_result_list)))
		  

if __name__ == "__main__":
	if not torch.cuda.is_available(): #
		print("Need available GPU(s) to run this model...") #
		quit() #

	main()
