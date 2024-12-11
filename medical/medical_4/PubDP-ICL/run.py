import numpy as np
import torch
import random
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from openicl import AccEvaluator, RougeEvaluator
from openicl import (
    DatasetReader,
    PromptTemplate,
    TopkRetriever,
    PPLInferencer,
    GenInferencer,
    PAGenInferencer,
)
from openicl import (
    TopkRetriever,
    ZeroRetriever,
    RandomRetriever,
    PateRetriever,
    BM25Retriever,
)
import openai
import os
import argparse

from gen_util import *
from tqdm import tqdm
import json
import asyncio

os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"

template_map = {
    "samsum": [
        '</E>Dialogue:"\n </dialogue>" \nThe summary is: </summary>',
        {"dialogue": "</dialogue>", "summary": "</summary"},
    ],
    "virattt/financial-qa-10K": [
        '</E>Context: "\n </context>"\n\nQuestion: </question>\nAnswer: </answer>',
        {"context": "</context>", "question": "</question>", "answer": "</answer>"},
    ],
    "Malikeh1375/medical-question-answering-datasets": [
        "</E>Question: </question>\n\nAnswer: </answer>\n",
        {"input": "</question>", "output": "</answer>"},
    ],
}
instruction_map = {
    "samsum": "Summarize the following dialogue:\n\n",
    "virattt/financial-qa-10K": None,
    "medical-question-answering-datasets": "You are a doctor, please answer the medical questions based on the patient's description.\n",
}
output_field_map = {
    "samsum": "summary",
    "virattt/financial-qa-10K": "answer",
    "Malikeh1375/medical-question-answering-datasets": "output",
}

# set all seeds
random.seed(0)
np.random.seed(0)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True


def train_test_split(dataset, ood_dataset=None):
    if not ood_dataset:
        if "test" not in dataset:
            if args.data_name == "Malikeh1375/medical-question-answering-datasets":
                dataset_sorted = sorted(
                    dataset["train"], key=lambda x: len(x["output"])
                )
                dataset = Dataset.from_list(dataset_sorted[3000:7000])
                dataset = dataset.train_test_split(test_size=100)
            else:
                dataset = dataset["train"].train_test_split(test_size=100)

        train_set, test_set = dataset["train"], dataset["test"]
        test_set = test_set.shuffle(seed=0).select(range(100))
        split_dataset = train_set.train_test_split(
            test_size=1 / 3
        )  # Split data into public and private
        private_dataset, public_dataset = (
            split_dataset["train"],
            split_dataset["test"],
        )

        return private_dataset, public_dataset, test_set

    else:
        if args.data_name == "Malikeh1375/medical-question-answering-datasets":
            dataset_sorted = sorted(dataset["train"], key=lambda x: len(x["output"]))
            dataset = Dataset.from_list(dataset_sorted[3000:7000])
            test_set = dataset.train_test_split(test_size=100)["test"]
            ood_dataset_sorted = sorted(
                ood_dataset["train"], key=lambda x: len(x["output"])
            )
            ood_dataset = Dataset.from_list(ood_dataset_sorted[7000:11000])
        test_set = test_set.shuffle(seed=0).select(range(100))

        return ood_dataset, test_set


async def main():
    template = PromptTemplate(
        template_map[args.data_name][0],
        template_map[args.data_name][1],
        ice_token="</E>",
    )
    # Output field name
    output_field = output_field_map[args.data_name]

    if not args.subset:
        dataset = load_dataset(args.data_name, cache_dir="./data")
    else:
        dataset = load_dataset(args.data_name, args.subset, cache_dir="./data")

    if args.ood:
        ood_dataset = load_dataset(args.data_name, args.ood_subset, cache_dir="./data")
    else:
        ood_dataset = None

    if not args.ood:
        private_dataset, public_dataset, testset = train_test_split(
            dataset, ood_dataset=ood_dataset
        )
        private_retriever = UWORetriever(private_dataset, testset, template)
        public_retriever = UWORetriever(public_dataset, testset, template)
        idx = private_retriever.retrieve(args.ice_num, args.ensemble, args.ds_size)
        idx_p = public_retriever.retrieve(args.ice_num, args.ensemble, args.ds_size)
        datasets = ["private", "public"]
        retriever = {"private": private_retriever, "public": public_retriever}
    else:
        public_dataset, testset = train_test_split(dataset, ood_dataset=ood_dataset)
        public_retriever = UWORetriever(public_dataset, testset, template)
        idx_p = public_retriever.retrieve(args.ice_num, args.ensemble, args.ds_size)
        retriever = {"public": public_retriever}
        datasets = ["public"]

    if "/" in args.data_name:
        args.data_name = args.data_name.split("/")[-1]

    if args.ood_subset:
        subset = args.ood_subset
        filename = f"output/ood_{args.data_name}_{subset}_{args.model_name}_{args.ensemble}way-{args.ice_num}shot"
    else:
        if args.subset:
            filename = f"output/{args.data_name}_{args.subset}_{args.model_name}_{args.ensemble}way-{args.ice_num}shot"
        else:
            filename = f"output/{args.data_name}_{args.model_name}_{args.ensemble}way-{args.ice_num}shot"
    print(filename)
    for dataset in datasets:
        print(f"#" * 15 + f"Generate {dataset} Predictions" + "#" * 15)
        predictions = []
        for i in tqdm(range(args.ds_size)):
            prompt_ensemble = retriever[dataset].prompt_generate(
                i, args.ice_num, args.ensemble, output_field=output_field
            )
            reference = testset[i][output_field]
            for j in tqdm(range(len(prompt_ensemble)), leave=False):
                if instruction_map[args.data_name]:
                    instruction = instruction_map[args.data_name]

                output = await complete(
                    prompt_ensemble[j],
                    args.max_token,
                    args.temp,
                    args.model_name,
                    instruction=instruction,
                )

                row = {
                    "Qidx": int(i),
                    "Eidx": int(j),
                    "prompt": prompt_ensemble[j],
                    "prediction": output,
                    "reference": reference,
                }
                predictions.append(row)
        with open(filename + f"_{dataset}.json", "w") as f:
            export_path = filename + f"_{dataset}.json"
            print(f"Export the output to {export_path}")
            json.dump(predictions, f, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="samsum")  # dataset name
parser.add_argument("--subset", type=str, default=None)  # subset name
parser.add_argument("--model_name", type=str, default="davinci-002")  # model name
parser.add_argument("--max_token", type=int, default=200)  # max token
parser.add_argument("--batch_size", type=int, default=1)  # batch size
# number of in-context examples
parser.add_argument("--ice_num", type=int, default=0)
parser.add_argument("--ensemble", type=int, default=1)  # number of ensemble
parser.add_argument("--ds_size", type=int, default=100)  # dataset size
parser.add_argument("--public_size", type=int, default=100)  # Public dataset size
parser.add_argument(
    "--output_json_filename", "--ojf", type=str, default="gpt2-xl_samsum_zero"
)  # output json filename
parser.add_argument("--temp", type=float, default=1.0)  # temperature
parser.add_argument("--nb", type=int, default=1)  # number of beams
parser.add_argument(
    "--ood",
    action="store_true",
    help="train and test set samples from different datasets",
)
parser.add_argument("--ood_subset", default=None)
args = parser.parse_args()
print(args)
# main()
asyncio.run(main())
