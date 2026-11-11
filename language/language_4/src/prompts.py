"""Prompt construction (chat template) and balanced option shuffling."""
import itertools

import numpy as np

from data_kobbq import ROLES

PERMS = list(itertools.permutations(range(3)))  # perm[k] = role index placed at letter k
LETTERS = "ABCD"


def assign_option_order(df, seed=42):
    """Seeded per-sample shuffle, balanced so unknown/stereo/anti occupy A/B/C equally.

    Within each (context_type, question_polarity) cell, samples are put in a seeded random
    order and the i-th sample gets permutation i % 6, so every role sits at every letter
    in 1/3 of the samples (up to rounding).
    """
    rng = np.random.RandomState(seed)
    perm_id = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby(["context_type", "question_polarity"]).indices.items():
        order = rng.permutation(idx)
        perm_id[order] = np.arange(len(order)) % len(PERMS)
    out = df.copy()
    out["perm_id"] = perm_id
    texts = out[["unknown_text", "stereo_text", "anti_text"]].values
    options, idx_cols = [], {r: [] for r in ROLES}
    for row_texts, p in zip(texts, perm_id):
        perm = PERMS[p]
        options.append([row_texts[r] for r in perm])
        for k, r in enumerate(perm):
            idx_cols[ROLES[r]].append(k)
    out["options"] = options
    for r in ROLES:
        out[f"{r}_option_idx"] = idx_cols[r]
    out["gold_idx"] = [row[f"{g}_option_idx"] for g, row in zip(out.gold_role, out.to_dict("records"))]
    return out


def kobbq_user_text(row, cfg):
    k = cfg["kobbq"]
    opts = "\n".join(f"{LETTERS[i]}: {o}" for i, o in enumerate(row["options"]))
    return (f"{k['instruction']}\n\n맥락: {row['context']}\n질문: {row['question']}\n{opts}\n\n"
            f"{k['answer_instruction']}")


def system_text(cfg, condition="base"):
    s = cfg["model"]["system_prompt"]
    if condition == "prompt":
        s = s + "\n" + cfg["steer"]["prompt_instruction"]
    return s


def chat(tok, system, turns, add_generation_prompt=True):
    """turns: list of (role, content). Returns the formatted string."""
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + \
           [{"role": r, "content": c} for r, c in turns]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


def kobbq_prompt(tok, row, cfg, condition="base", answer_letter=None):
    """If answer_letter is given, the assistant turn starts with that letter (CAA)."""
    text = chat(tok, system_text(cfg, condition), [("user", kobbq_user_text(row, cfg))])
    if answer_letter is not None:
        text = text + answer_letter
    return text


def kmmlu_block(q):
    return f"{str(q['question']).strip()}\nA. {q['A']}\nB. {q['B']}\nC. {q['C']}\nD. {q['D']}"


def kmmlu_prompt(tok, row, shots, cfg, condition="base"):
    """Multi-turn k-shot: dev examples as (user question, assistant letter) pairs."""
    head = cfg["kmmlu"]["instruction"].format(subject=row["subject"].replace("-", " ")) + "\n\n"
    turns = []
    for i, s in enumerate(shots):
        turns.append(("user", (head if i == 0 else "") + kmmlu_block(s)))
        turns.append(("assistant", LETTERS[int(s["answer"]) - 1]))
    turns.append(("user", (head if not shots else "") + kmmlu_block(row)))
    return chat(tok, system_text(cfg, condition), turns)


def benign_prompt(tok, text, cfg):
    return chat(tok, system_text(cfg, "base"), [("user", text)])


def kmmlu_prefix_suffix(tok, row, shots, cfg, condition="base"):
    """Split the k-shot prompt into the shared (system + shots) prefix and the per-question suffix."""
    full = kmmlu_prompt(tok, row, shots, cfg, condition)
    marker = "<|im_start|>user\n" + (cfg["kmmlu"]["instruction"].format(subject=row["subject"].replace("-", " "))
                                     + "\n\n" if not shots else "") + kmmlu_block(row)
    cut = full.rindex(marker)
    return full[:cut], full[cut:]
