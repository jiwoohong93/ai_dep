import random
import os
import json
import openai
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type
from tqdm import tqdm
import logging
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import asyncio
import aiohttp

# from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.getenv("OPENAI_API_KEY")
# CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", False))


class EntailmentLLM:
    def __init__(self, model):
        self.model = model
        self.max_tokens = 100
        self.temp = 0.02

    def equivalence_prompt(self, text1, text2, question=None):
        if question:
            prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
            prompt += "Here are two possible answers:\n"
            prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
            prompt += (
                "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."
                ""
            )
        else:
            prompt = f"""We are evaluating summaries \n"""
            prompt += "Here are two possible summaries:\n"
            prompt += f"Possible Summary 1: {text1}\nPossible Summary 2: {text2}\n"
            prompt += (
                "Does Possible Summary 1 semantically entail Possible Summary 2? Respond with entailment, contradiction, or neutral."
                ""
            )

        return prompt

    def check_implication(self, text1, text2, question):
        prompt = self.equivalence_prompt(text1, text2, question)
        binary_response = predict(
            prompt, temperature=self.temp, model=self.model
        ).lower()
        if "entailment" in binary_response:
            return 2
        elif "neutral" in binary_response:
            return 1
        elif "contradiction" in binary_response:
            return 0
        else:
            logging.warning("MANUAL NEUTRAL!")
            return 1


def predict(prompt, temperature=1.0, model="gpt-4"):
    """Predict with GPT models."""

    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    if model == "gpt-4":
        model = "gpt-4"
    elif model == "gpt-4-turbo":
        model = "gpt-4-1106-preview"
    elif model == "gpt-3.5":
        model = "gpt-3.5-turbo-1106"

    output = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )

    response = output.choices[0].message.content
    return response


async def get_embedding(text, placeholder, model=None):
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    data = {"input": text, "model": model}

    url = "https://api.openai.com/v1/embeddings"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = result["data"][0]["embedding"]
                    return embedding
                else:
                    print(f"Error {response.status}: {await response.text()}")
                    print(f"Cannot embed {text}, returning placeholder")
                    return placeholder
        except Exception as e:
            print(f"Exception occurred: {e}")
            print(f"Cannot embed {text}, returning placeholder")
            return placeholder


def get_semantic_ids(strings_list, model, strict_entailment=False, question=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2, question):
        implication_1 = model.check_implication(text1, text2, question=question)
        implication_2 = model.check_implication(
            text2, text1, question=question
        )  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            """
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and (
                [1, 1] != implications
            )
            """
            # Modified entailment rule
            implications = [implication_1, implication_2]
            semantically_equivalent = (0 not in implications) and (2 in implications)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in tqdm(enumerate(strings_list)):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j], question):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids, strings_list


async def construct_semantic_groups(
    strings_list, model, similarity_threshold=0.8, only_centroid=False, path=None
):
    """
    Constructs semantic groups using an adaptive k-nearest neighbors approach based on a similarity threshold.

    Parameters:
    embeddings (numpy array): Semantic embeddings of shape (n_samples, embedding_dim)
    similarity_threshold (float): Minimum cosine similarity required to group sentences (default: 0.8)

    Returns:
    List of lists: Each sublist contains indices of sentences that belong to the same group.
    """
    # Step 1: Compute cosine similarity matrix

    try:
        placeholder = (
            openai.Embedding.create(input=[strings_list[0][0]], model=model)
            .data[0]
            .embedding
        )  # 여기서도 운 나쁘면 error 발생
    except openai.error.InvalidRequestError as e:
        placeholder = (
            openai.Embedding.create(input=[strings_list[1][0]], model=model)
            .data[0]
            .embedding
        )

    embeddings = []
    public_indices = []
    for i, text in enumerate(strings_list):
        embedding = await get_embedding(text[0], placeholder, model=model)
        embeddings.append(embedding)
        if "public" in text[1]:
            public_indices.append(i)

    if path:
        with open(npy_path, "wb") as f:
            np.save(f, np.array(embeddings))

    # embeddings = [emb for emb in embeddings if emb is not None]  # filter missing value
    embeddings = np.array(
        embeddings
    )  # shape should be [public_private, openai.embedding.dimension]

    similarity_matrix = cosine_similarity(embeddings)

    # Step 2: Initialize group assignments (-1 means unassigned)
    n = embeddings.shape[0]
    groups = [-1] * n
    cluster_embeddings = {}
    current_group_id = 0
    cluster = {}
    # Step 3: Create groups based on adaptive nearest neighbors
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

    public_cluster = {}
    public_embeddings = {}

    for key, value in cluster.items():
        # Filter values based on public_indices
        filtered_values = [v for v in value if v in public_indices]
        public_cluster[key] = (
            filtered_values if filtered_values else value
        )  # Fallback to original value if filtered list is empty

        # Generate public_embeddings for the key
        public_embeddings[key] = [embeddings[v] for v in public_cluster[key]]

    # Record the histogram count
    count = [len(value) for value in cluster.values()]
    # Find Representatives
    for k, v in cluster.items():
        if len(cluster[k]) > 1:
            centroid = np.mean(cluster_embeddings[k], axis=0)
            distances_to_centroid = np.linalg.norm(
                public_embeddings[k] - centroid, axis=1
            )
            closest_idx = public_cluster[k][np.argmin(distances_to_centroid)]
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


def load_ensemble_ouptut(path, query_extract=False):
    with open(path, "r") as f:
        file = json.load(f)
        ensemble_output = []
        ensemble_query = []
        references = []
        for line in file:
            qidx, edix, prompt, prediction = (
                int(line["Qidx"]),
                int(line["Eidx"]),
                line["prompt"],
                line["prediction"],
            )
            if edix == 0:
                ensemble_output.append([prediction])
                if query_extract:
                    query = prompt.split("\n")[-2].split(":")[-1].strip()  # not sure
                    ensemble_query.append(query)
                reference = line["reference"]
                references.append(references)
            else:
                ensemble_output[qidx].append(prediction)

    return ensemble_output, ensemble_query, references


def convert_to_mm_ss(epoch_seconds):
    minutes = int(epoch_seconds // 60)  # Integer division for minutes
    seconds = int(epoch_seconds % 60)  # Modulo operation for remaining seconds

    # Format the result as mm:ss
    formatted_time = f"{minutes:02}:{seconds:02}"

    return formatted_time


def load_ensembles(path, tag, query_parser=None, gamma=1.0):
    with open(path, "r") as f:
        file = json.load(f)
        ensembles = {}
        queries = []  # Quries can be dialogue or question
        references = []

        for i, line in enumerate(file):
            qidx, eidx, prompt, prediction = (
                int(line["Qidx"]),
                int(line["Eidx"]),
                line["prompt"],
                line["prediction"],
            )
            prediction = [prediction, tag]  # e.g., prediction -> [prediction, 'public']
            if eidx == 0:
                ensembles[qidx] = [prediction]
                if query_parser == "QA":
                    query = prompt.split("Question:")[-1].split("\n")[0].strip()
                elif query_parser == "summary":
                    query = prompt.split("Dialogue:")[-1].split("Summarize")[0].strip()
                else:
                    raise NotImplementedError("Support query_parser='QA' or 'summary'")
                queries.append(query)
                references.append(line["reference"])
            else:
                ensembles[qidx].append(prediction)

                if i % 100 == 99 and gamma < 1.0:
                    ensembles[qidx] = random.sample(
                        ensembles[qidx], int(len(ensembles[qidx]) * gamma)
                    )
    return ensembles, queries, references


def main():
    private_path = args.path + "_private.json"
    if args.ood:
        assert args.ood_path
        public_path = args.ood_path + "_public.json"
    else:
        public_path = args.path + "_public.json"
    private_ensembles, queries, references = load_ensembles(
        private_path, "private", query_parser=args.query_type, gamma=args.gamma
    )
    public_ensembles, _, _ = load_ensembles(
        public_path, "public", query_parser=args.query_type, gamma=args.gamma
    )

    ensembles = {}
    for i in range(len(private_ensembles)):
        if args.aggregate:
            ensembles[i] = private_ensembles[i] + public_ensembles[i]
        else:
            ensembles[i] = private_ensembles[i]

    start = time.time()
    export_output = []
    if args.aggregate:
        filename = args.path.split("/")[-1] + "_aggregate.json"
    else:
        filename = args.path.split("/")[-1] + "_private.json"
    if args.ood:
        path = f"semantic_group/oodpublic_{args.similarity_threshold}_{filename}"
    else:
        path = f"semantic_group/{args.gamma}_{args.similarity_threshold}_{filename}"
    print(f"\nExport path: {path}\n")

    if args.export_embedding:
        npy_subpath = filename.split(".")[0]
        if not os.path.exists(f"semantic_group/{npy_subpath}"):
            os.mkdir(f"semantic_group/{npy_subpath}")
    else:
        npy_subpath = None

    if not args.debug:
        num_examples = len(ensembles)
    else:
        num_examples = 1

    only_centroid = not args.debug
    for i in tqdm(range(num_examples)):
        # npy_path = f"semantic_group/{npy_subpath}/{i}.npy"
        semantic_group, count = asyncio.run(
            construct_semantic_groups(
                ensembles[i],
                args.embedding_model,
                similarity_threshold=args.similarity_threshold,
                only_centroid=only_centroid,
                path=npy_subpath,
            )
        )
        semantic_group["count"] = count
        semantic_group["query"] = queries[i]
        semantic_group["reference"] = references[i]
        export_output.append(semantic_group)
    end = time.time()
    print(f"Elapsed time: {convert_to_mm_ss(end-start)}\n\n")
    with open(path, "w") as f:
        json.dump(export_output, f, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="Dataset path")
parser.add_argument("--ood", action="store_true")
parser.add_argument(
    "--ood_path", type=str, default=None, help="path for OOD public data"
)
parser.add_argument(
    "--embedding_model",
    type=str,
    default="text-embedding-ada-002",
    help="Embedding model name",
)
parser.add_argument(
    "--query_type", type=str, required=True, help="Query type (QA or summary)"
)
parser.add_argument(
    "--aggregate", action="store_true", help="Aggregate private and public ensembles"
)
parser.add_argument("--export_embedding", action="store_true", help="Export embeddings")
parser.add_argument(
    "--similarity_threshold", type=float, default=0.90, help="Similarity threshold"
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="If debug is flagged, only run for 1 ensemble group",
)
parser.add_argument("--gamma", type=float, default=1.0, help="Fraction of ensembels")

args = parser.parse_args()
print(args)
main()
