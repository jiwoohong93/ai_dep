#!/usr/bin/env python
# coding: utf-8

# In[1]:

import heapq
from datasets import load_dataset, Dataset
import openai
import random
import numpy as np
from openicl import PromptTemplate
from gen_util import *
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import json
import argparse


load_config = {
    "dataset": "Malikeh1375/medical-question-answering-datasets",
    "subset": "chatdoctor_healthcaremagic",
}
load_config2 = {
    "dataset": "Malikeh1375/medical-question-answering-datasets",
    "subset": "chatdoctor_icliniq",
}
os.environ["OPENAI_API_KEY"] = "YOU OPENAI API KEY"


template_map = {
    "samsum": [
        '</E>Dialogue:"\n </dialogue>" \nSummarize the above dialogue: </summary>',
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
data_name = "Malikeh1375/medical-question-answering-datasets"
template = PromptTemplate(
    template_map[data_name][0],
    template_map[data_name][1],
    ice_token="</E>",
)


complete_config = {
    "l": 200,
    "temp": 0.02,
    "model_name": "gpt-3.5-turbo",
    "embedding_model": "text-embedding-ada-002",
}


def first_sentence(output):
    words = output.split(".")[0]
    return words


def first_five_words(sentence):
    words = sentence.split()
    return " ".join(words[:6])


def generate_demonstration(indices, members, query, template):
    prompt_temp = ""
    for j in range(len(indices)):
        idx = indices[j]  # ith ensemble jth ICE
        output = template.generate_item(members[idx])
        if j == 0:
            prompt_temp += output
        else:
            prompt_temp += "\n" + output
    prompt_temp += "\n" + query
    return prompt_temp


def complete_api(prompt, l, temp, model_name, num_log_probs=None, echo=False, n=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    while not received:
        try:
            if "turbo" in model_name or "4o" in model_name:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
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


# ### SGA

# In[6]:


from semantic_group.jointEM import *


def ensemble_infernece(indices, members, query, template):
    predictions = []
    for idx in indices:
        prompt = generate_demonstration(idx, members, query, template)
        response = complete_api(
            prompt, complete_config["l"], 0.8, complete_config["model_name"]
        )
        predictions.append(response)
    return predictions


def get_embedding(text, model=None, sub_model=None):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    try:
        embedding = openai.Embedding.create(input=[text], model=model).data[0].embedding
    except:
        embedding = (
            openai.Embedding.create(input=[text], model=sub_model).data[0].embedding
        )

    return embedding


def construct_semantic_groups(
    strings_list, model, similarity_threshold=0.8, only_centroid=False, path=None
):
    # Step 1: Compute cosine similarity matrix
    embeddings = [
        get_embedding(text, model=model, sub_model="text-embedding-3-small")
        for text in strings_list
    ]
    embeddings = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings)

    # Step 2: Initialize group assignments (-1 means unassigned)
    n = embeddings.shape[0]
    groups = [-1] * n
    cluster_embeddings = {}
    cluster = {}
    current_group_id = 0

    # Step 3: Create groups
    for i in range(n):
        if groups[i] == -1:  # If sentence i is not assigned to a group
            groups[i] = current_group_id  # Assign a new group ID to sentence i
            cluster[current_group_id] = [i]
            cluster_embeddings[current_group_id] = np.expand_dims(
                np.array(embeddings[i]), axis=0
            )
            # Find neighbors with similarity greater than the threshold
            for j in range(i + 1, n):
                if groups[j] != -1:  # if jth element is already assigned, then pass
                    continue
                else:
                    if similarity_matrix[i][j] >= similarity_threshold:
                        groups[j] = (
                            current_group_id  # Assign the same group if similarity is above the threshold
                        )
                        cluster[current_group_id].append(j)
                        cge = cluster_embeddings[current_group_id]
                        new_embed = np.expand_dims(embeddings[j], axis=0)
                        cluster_embeddings[current_group_id] = np.concatenate(
                            (cge, new_embed), axis=0
                        )
            current_group_id += 1
        else:
            continue

    # Record the histogram count
    count = [len(value) for value in cluster.values()]
    # Find Representatives
    for k, v in cluster.items():
        if len(cluster[k]) > 1:
            centroid = np.mean(cluster_embeddings[k], axis=0)
            distances_to_centroid = np.linalg.norm(
                cluster_embeddings[k] - centroid, axis=1
            )
            closest_idx = cluster[k][np.argmin(distances_to_centroid)]
        else:
            closest_idx = cluster[k][0]
        cluster[k] = closest_idx

    # Step 4: Organize sentences into their respective groups
    # Return the groups as a list of lists (each list contains sentence indices belonging to the same group)
    if only_centroid:
        semantic_group = {}
        for k, v in cluster.items():
            semantic_group[k] = [strings_list[cluster[k]]]
        return semantic_group, count
    else:

        def semantic_group_embedding(semantic_ids, strings):
            num_group = max(semantic_ids) + 1
            semantic_group = {}
            # Initialize
            for i in range(num_group):
                semantic_group[i] = []
            for sid, string in zip(semantic_ids, strings):
                semantic_group[sid].append(string)

            return semantic_group

        semantic_group = semantic_group_embedding(list(groups), strings_list)
        for k, v in cluster.items():
            semantic_group[k].append({"Representative": [strings_list[cluster[k]]]})

        return semantic_group, count


def selectwith_public1shot(public_example, query, candidates):
    prompt = "["
    for candidate in candidates:
        prompt += candidate + ",\n"
    prompt = prompt[:-2] + "]"
    prompt = prompt + "\nThe answer is:"
    query = (
        query
        + "\nPick the most accurate answer for the question with the following answer candidates ranked by their freqeuncy from high to low: "
    )
    context = public_example + "Question:\n" + query
    final_prompt = context + prompt

    return final_prompt


def top_k_indices(counts, k):
    # Get the indices of the k largest elements
    return heapq.nlargest(k, range(len(counts)), key=lambda i: counts[i])


def main2():
    icliniq = load_dataset(
        load_config2["dataset"], load_config["subset"], cache_dir="./data"
    )
    dataset = icliniq["train"]
    dataset_sorted = sorted(dataset, key=lambda x: len(x["output"]))
    dataset = Dataset.from_list(dataset_sorted[3000:7000])
    random.seed(0)
    np.random.seed(0)
    if args.balance:
        dataset = dataset.train_test_split(test_size=120)
        dataset = dataset["test"].train_test_split(test_size=80)
        private, public = dataset["test"], dataset["train"]
        private = private.train_test_split(test_size=40)
        members, nonmembers = private["train"], private["test"]

    else:
        dataset = dataset.train_test_split(test_size=240)
        dataset = dataset["test"].train_test_split(test_size=200)
        private, public = dataset["test"], dataset["train"]
        private = private.train_test_split(test_size=160)
        members, nonmembers = private["train"], private["test"]

    member_query = [
        f"Complete the following question\n Question: {first_five_words(member)}"
        for member in members["input"]
    ]
    nonmember_query = [
        f"Complete the following question\n Question: {first_five_words(member)}"
        for member in nonmembers["input"]
    ]

    # In[89]:

    retriever1 = UWORetriever(members, member_query, template, seed=0)
    idx_member = retriever1.retrieve(20, 1, 40)[0]
    retriever2 = UWORetriever(members, nonmember_query, template, seed=0)
    idx_nonmember = retriever2.retrieve(20, 1, 160)[0]

    # In[6]:

    member_sims = []
    completions = []
    references = []
    for i, member in enumerate(tqdm(members)):
        prompt = generate_demonstration(
            idx_member[i], members, member_query[i], template
        )
        reference = members[i]["input"]
        response = complete_api(
            prompt, complete_config["l"], 0.02, complete_config["model_name"]
        )
        model_completion = (member_query[i] + response).split("Question:")[-1]
        completions.append(model_completion)
        references.append(reference)
        completion_embedding = (
            openai.Embedding.create(
                input=[model_completion], model=complete_config["embedding_model"]
            )
            .data[0]
            .embedding
        )
        reference_embedding = (
            openai.Embedding.create(
                input=[reference], model=complete_config["embedding_model"]
            )
            .data[0]
            .embedding
        )
        similarity = cosine_similarity([completion_embedding], [reference_embedding])
        member_sims.append(float(similarity[0].item()))

    with open("MIA/output_2shot_10way_nonprivate.json", "w") as f:
        examples = [
            {"Completion": c, "Reference": r, "Score": s}
            for (c, r, s) in zip(completions, references, member_sims)
        ]
        json.dump(examples, f, indent=4)

    nonmember_sims = []
    for i, member in enumerate(tqdm(nonmembers)):
        prompt = generate_demonstration(
            idx_nonmember[i], members, nonmember_query[i], template
        )
        reference = nonmembers[i]["input"]
        response = complete_api(
            prompt, complete_config["l"], 0.02, complete_config["model_name"]
        )
        model_completion = (nonmember_query[i] + response).split("Question:")[-1]
        completion_embedding = (
            openai.Embedding.create(
                input=[model_completion], model=complete_config["embedding_model"]
            )
            .data[0]
            .embedding
        )
        reference_embedding = (
            openai.Embedding.create(
                input=[reference], model=complete_config["embedding_model"]
            )
            .data[0]
            .embedding
        )
        similarity = cosine_similarity([completion_embedding], [reference_embedding])
        nonmember_sims.append(float(similarity[0]))

    labels = np.concatenate([np.ones(len(member_sims)), np.zeros(len(nonmember_sims))])
    scores = np.concatenate([member_sims, nonmember_sims])
    auroc = roc_auc_score(labels, scores)
    auroc
    print(auroc)


def main():
    icliniq = load_dataset(
        load_config2["dataset"], load_config["subset"], cache_dir="./data"
    )
    dataset = icliniq["train"]
    dataset_sorted = sorted(dataset, key=lambda x: len(x["output"]))
    dataset = Dataset.from_list(dataset_sorted[3000:7000])
    random.seed(0)
    np.random.seed(0)
    if args.balance:
        dataset = dataset.train_test_split(test_size=120)
        dataset = dataset["test"].train_test_split(test_size=80)
        private, public = dataset["test"], dataset["train"]
        private = private.train_test_split(test_size=40)
        members, nonmembers = private["train"], private["test"]
        # Balanced Scenario
        eps_1 = 0.093
        eps_3 = 0.216
        eps_8 = 0.537

    else:
        dataset = dataset.train_test_split(test_size=240)
        dataset = dataset["test"].train_test_split(test_size=200)
        private, public = dataset["test"], dataset["train"]
        private = private.train_test_split(test_size=160)
        members, nonmembers = private["train"], private["test"]
        # Unbalanced Scenario
        eps_1 = 0.057
        eps_3 = 0.137
        eps_8 = 0.339

    member_query = [
        f"Complete the follwing question.\n Question: {first_five_words(member)}"
        for member in members["input"]
    ]
    nonmember_query = [
        f"Complete the follwing question.\n Question: {first_five_words(member)}"
        for member in nonmembers["input"]
    ]

    ice_num = 2
    num_ensemble = 10
    retriever = UWORetriever(members, member_query, template, seed=0)
    public_retriever = UWORetriever(public, member_query, template, seed=0)
    idx_member = retriever.retrieve(ice_num, num_ensemble, len(members))
    idx_nonmember = retriever.retrieve(ice_num, num_ensemble, len(nonmembers))
    idx_public_member = public_retriever.retrieve(ice_num, num_ensemble, len(members))
    idx_public_nonmember = public_retriever.retrieve(
        ice_num, num_ensemble, len(nonmembers)
    )

    member_sims = []
    trials = 1
    eps = np.inf
    k = 6

    completions = []
    references = []
    export_path = f"MIA/output_2shot_10way_{str(eps)}.json"
    for i, _ in enumerate(tqdm(members)):
        # DP part
        reference = members[i]["input"]
        idx_i = [idx[i] for idx in idx_member]
        idx_p = [idx[i] for idx in idx_public_nonmember]
        predictions_pri = ensemble_infernece(idx_i, members, member_query[i], template)
        predictions_pub = ensemble_infernece(idx_p, public, member_query[i], template)
        predictions = predictions_pri + predictions_pub
        semantic_group, count = construct_semantic_groups(
            predictions,
            complete_config["embedding_model"],
            similarity_threshold=0.92,
            only_centroid=True,
        )
        if eps == np.inf:
            joint_out = top_k_indices(np.array(count), min(k, len(count)))
        else:
            joint_out = joint(np.array(count), min(k, len(count)), eps, neighbor_type=1)
        candidates = []
        for idx in joint_out:
            candidates.append(
                list(semantic_group.values())[idx][0]
            )  # pick representative
        public_example = template.generate_item(public[1])
        final_prompt = selectwith_public1shot(
            public_example, member_query[i], candidates
        )
        # Attack part
        response = complete_api(
            final_prompt,
            complete_config["l"],
            complete_config["temp"],
            complete_config["model_name"],
        )
        model_completion = (member_query[i] + response).split("Question:")[-1]
        references.append(reference)
        completions.append(model_completion)
        completion_embedding = (
            openai.Embedding.create(
                input=[model_completion], model=complete_config["embedding_model"]
            )
            .data[0]
            .embedding
        )
        reference_embedding = (
            openai.Embedding.create(
                input=[reference], model=complete_config["embedding_model"]
            )
            .data[0]
            .embedding
        )
        similarity = cosine_similarity([completion_embedding], [reference_embedding])
        member_sims.append(float(similarity[0].item()))

    with open(export_path, "w") as f:
        examples = [
            {"Completion": c, "Reference": r, "Score": s}
            for (c, r, s) in zip(completions, references, member_sims)
        ]
        json.dump(examples, f, indent=4)

    nonmember_sims = []
    for i, _ in enumerate(tqdm(nonmembers)):
        # DP part
        reference = nonmembers[i]["input"]
        idx_i = [idx[i] for idx in idx_nonmember]
        idx_p = [idx[i] for idx in idx_public_nonmember]
        predictions_pri = ensemble_infernece(
            idx_i, members, nonmember_query[i], template
        )
        predictions_pub = ensemble_infernece(
            idx_p, public, nonmember_query[i], template
        )
        predictions = predictions_pri + predictions_pub
        semantic_group, count = construct_semantic_groups(
            predictions,
            complete_config["embedding_model"],
            similarity_threshold=0.92,
            only_centroid=True,
        )
        if eps == np.inf:
            joint_out = top_k_indices(np.array(count), min(k, len(count)))
        else:
            joint_out = joint(np.array(count), min(k, len(count)), eps, neighbor_type=1)
        candidates = []
        for idx in joint_out:
            candidates.append(
                list(semantic_group.values())[idx][0]
            )  # pick representative
        prompt = template.generate_item(public[1])
        final_prompt = selectwith_public1shot(prompt, nonmember_query[i], candidates)
        # Attack part
        response = complete_api(
            final_prompt,
            complete_config["l"],
            complete_config["temp"],
            complete_config["model_name"],
        )
        model_completion = (nonmember_query[i] + response).split("Question:")[-1]
        completion_embedding = (
            openai.Embedding.create(
                input=[model_completion], model=complete_config["embedding_model"]
            )
            .data[0]
            .embedding
        )
        reference_embedding = (
            openai.Embedding.create(
                input=[reference], model=complete_config["embedding_model"]
            )
            .data[0]
            .embedding
        )
        similarity = cosine_similarity([completion_embedding], [reference_embedding])
        nonmember_sims.append(float(similarity[0].item()))

    labels = np.concatenate([np.ones(len(member_sims)), np.zeros(len(nonmember_sims))])
    scores = np.concatenate([member_sims, nonmember_sims])
    auroc = roc_auc_score(labels, scores)
    auroc
    print(f"eps={eps}, k={k}, balance={args.balance}")
    print(auroc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIA attack")
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balnced Setting",
    )
    args = parser.parse_args()
    main()
