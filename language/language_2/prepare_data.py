#!/usr/bin/env python3
"""KOLD 데이터를 실험용 80/10/10 split 으로 나눈다.

- 하나의 전역 split 을 만들고 각 task 는 그 부분집합을 사용한다 (task 간 test 누수 방지).
  stratify 키: OFF / TGT / Level C 속성.
- Level B (target type): OFF=True 샘플, 4 클래스 (UNT/IND/OTH/GRP)
- Level C: TGT=group 샘플. GRP 가 여러 개(" & ")면 첫 라벨을 쓴다 (language_1 검증과 동일한 기준).
    grp      : target group 속성 5 클래스 (GEN/RACE/POL/REL/MISC)
    grp_fine : 세부 target group (예: GEN-Women, RACE-Others), 논문 Table 3/10
- Span: 어노테이터 3명 중 2명 이상이 표시한 문자 (끝 인덱스 inclusive).
  공개 OFF_span/TGT_span 문자열과 99.9% 일치한다.

실행:
    python language_2/prepare_data.py [--data PATH] [--out DIR] [--seed 42]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "KOLD" / "data" / "kold_v1.json"
DEFAULT_OUT = HERE / "data"

TGT_MAP = {"untargeted": "UNT", "individual": "IND", "other": "OTH", "group": "GRP"}

# GRP 라벨 -> (Level C 속성, 세부 target group). 논문 Table 3/10 기준, language_1/validate_kold.py 와 동일.
GROUP_MAP = {
    "gender-LGBTQ+": ("GEN", "LGBTQ+"),
    "gender-queer": ("GEN", "LGBTQ+"),
    "sexual_orientation-homosexual": ("GEN", "LGBTQ+"),
    "sexual_orientation-others": ("GEN", "LGBTQ+"),
    "gender-female": ("GEN", "Women"),
    "gender-male": ("GEN", "Men"),
    "gender-others": ("GEN", "Others"),
    "race-asian": ("RACE", "Asian"),
    "race-black": ("RACE", "Black"),
    "race-chinese": ("RACE", "Chinese"),
    "race-indian": ("RACE", "Indian"),
    "race-korean_chinese": ("RACE", "Korean-Chinese"),
    "race-southeast_asian": ("RACE", "Southeast Asian"),
    "race-white": ("RACE", "White"),
    "race-others": ("RACE", "Others"),
    "politics-conservative": ("POL", "Conservative"),
    "politics-progressive": ("POL", "Progressive"),
    "politics-others": ("POL", "Others"),
    "religion-buddhism": ("REL", "Buddhism"),
    "religion-catholic": ("REL", "Catholic"),
    "religion-christian": ("REL", "Christian"),
    "religion-islam": ("REL", "Islam"),
    "religion-others": ("REL", "Others"),
    "gender-feminist": ("MISC", "Feminist"),
    "others-feminist": ("MISC", "Feminist"),
    "others-age": ("MISC", "Agism"),
    "others-disability": ("MISC", "Disabled"),
    "others-disease": ("MISC", "Diseased"),
    "others-physical_appearance": ("MISC", "Physical Appearance"),
    "others-socioeconomic_status": ("MISC", "Socio-economic Status"),
    "others-others": ("MISC", "Others"),
}


def majority_spans(comment, raw_labels, start_key, end_key, min_votes=2):
    """어노테이터 span 을 문자 단위로 투표해 [start, end) 구간 리스트로 반환."""
    votes = [0] * len(comment)
    for r in raw_labels:
        for s, e in zip(r.get(start_key) or [], r.get(end_key) or []):
            for k in range(max(0, s), min(e + 1, len(comment))):
                votes[k] += 1
    spans, i = [], 0
    while i < len(comment):
        if votes[i] >= min_votes:
            j = i
            while j < len(comment) and votes[j] >= min_votes:
                j += 1
            spans.append([i, j])
            i = j
        else:
            i += 1
    return spans


def convert(x):
    tgt = TGT_MAP.get(x["TGT"])
    first_grp = x["GRP"].split(" & ")[0] if tgt == "GRP" else None
    off_spans = majority_spans(x["comment"], x["raw_labels"], "off_start_idx", "off_end_idx")
    tgt_spans = majority_spans(x["comment"], x["raw_labels"], "tgt_start_idx", "tgt_end_idx")
    # 공개 라벨에서 span 이 비어 있으면 재구성 결과도 비운다 (불일치 수십 건 처리)
    if not x["OFF_span"]:
        off_spans = []
    if not x["TGT_span"]:
        tgt_spans = []
    return {
        "guid": x["guid"],
        "title": x["title"],
        "comment": x["comment"],
        "off": int(x["OFF"]),
        "tgt": tgt,
        "grp": GROUP_MAP[first_grp][0] if first_grp else None,
        "grp_fine": "-".join(GROUP_MAP[first_grp]) if first_grp else None,
        "off_spans": off_spans,
        "tgt_spans": tgt_spans,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [convert(x) for x in json.load(open(args.data, encoding="utf-8"))]
    strat = [f"{r['off']}|{r['tgt']}|{r['grp']}" for r in rows]
    train, rest, _, strat_rest = train_test_split(
        rows, strat, test_size=0.2, stratify=strat, random_state=args.seed)
    dev, test = train_test_split(
        rest, test_size=0.5, stratify=strat_rest, random_state=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("dev", dev), ("test", test)]:
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(split),
              "OFF", sum(r["off"] for r in split),
              "TGT", dict(Counter(r["tgt"] for r in split if r["tgt"])),
              "GRP", dict(Counter(r["grp"] for r in split if r["grp"])),
              "n_fine", len({r["grp_fine"] for r in split if r["grp_fine"]}))


if __name__ == "__main__":
    main()
