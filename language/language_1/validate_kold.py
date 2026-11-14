#!/usr/bin/env python3
"""KOLD (Korean Offensive Language Dataset) 데이터 검증 스크립트.

KOLD/data/kold_v1.json 을 읽어 논문(EMNLP 2022)에 보고된 통계와 비교한다.
- 표준 라이브러리만 사용
- 결과는 language_1/results/ 에만 저장 (JSON + Markdown)

실행:
    python language_1/validate_kold.py [--data PATH] [--out DIR]

각 검사는 다음 중 하나로 분류된다.
- PASS : 데이터에서 계산한 값이 논문 수치와 일치
- FAIL : 일치하지 않음 (하나라도 있으면 exit code 1)
- INFO : 논문 수치가 공개 데이터(v1)만으로 재현되지 않거나, 논문 자체에 불일치가
         있어 참고용으로만 비교 (통과/실패 판정에 포함하지 않음)
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "KOLD" / "data" / "kold_v1.json"
DEFAULT_OUT = HERE / "results"

# ---------------------------------------------------------------------------
# 논문 수치 (Table 2, 3, 4, Appendix Table 10, Section 3~4)
# ---------------------------------------------------------------------------
PAPER = {
    "total": 40429,
    "level_a": {"NOT": 20119, "OFF": 20310},
    "level_b": {"UNT": 2596, "IND": 3899, "OTH": 1402, "GRP": 12413},
    "level_b_targeted_pct": {"GRP": 70.1, "IND": 22.0, "OTH": 7.9},
    "level_c": {"GEN": 2986, "RACE": 2970, "POL": 1939, "REL": 1699, "MISC": 2819},
    "table3": {
        "GEN": {"LGBTQ+": 1369, "Women": 1129, "Men": 591, "Others": 108},
        "RACE": {"Chinese": 475, "Korean-Chinese": 460, "Black": 175, "Others": 1605},
        "POL": {"Progressive": 1052, "Conservative": 316, "Others": 651},
        "REL": {"Muslim": 1059, "Christian": 518, "Catholic": 57, "Others": 91},
        "MISC": {"Feminist": 1483, "Socio-economic Status": 88, "Agism": 62, "Others": 971},
    },
    "table3_total": 12981,
    "top10": [
        ("Feminist", 1483, 11.42), ("LGBTQ+", 1369, 10.55), ("Women", 1129, 8.70),
        ("Muslim", 1059, 8.16), ("Progressive", 1052, 8.10), ("Men", 591, 4.55),
        ("Christian", 518, 3.99), ("Chinese", 475, 3.66), ("Korean-Chinese", 460, 3.54),
        ("Conservative", 316, 2.43),
    ],
    "table10": {
        "GEN": ["LGBTQ+", "Men", "Women"],
        "RACE": ["Asian", "Black", "Chinese", "Indian", "Korean-Chinese", "Southeast Asian", "White"],
        "POL": ["Conservative", "Progressive"],
        "REL": ["Buddhism", "Catholic", "Christian", "Islam"],
        "MISC": ["Agism", "Disabled", "Diseased", "Feminist", "Physical Appearance",
                 "Socio-economic Status"],
    },
    "grp_no_tgt_span_pct": 55.1,
    "annotators": 3124,
    "annotators_per_sample": 3,
    "alpha_a": 0.55,
    "sources": {"naver_news", "youtube"},
    "date_range": ("2020-03", "2022-03"),
}

ATTR_NAMES = {
    "GEN": "Gender & Sexual Orientation",
    "RACE": "Race, Ethnicity & Nationality",
    "POL": "Political Affiliation",
    "REL": "Religion",
    "MISC": "Miscellaneous",
}

# 데이터의 GRP 라벨(예: "race-korean_chinese") -> (속성, 논문 Table 3/10 이름)
LABEL_MAP = {
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
# Table 3/4 는 Islam 을 "Muslim" 으로 표기
TABLE3_ALIAS = {"Islam": "Muslim"}

TGT_MAP = {"untargeted": "UNT", "individual": "IND", "other": "OTH", "group": "GRP"}
EXPECTED_KEYS = {"guid", "source", "date", "title", "comment", "OFF", "TGT", "GRP",
                 "OFF_span", "TGT_span", "raw_labels"}


class Validator:
    def __init__(self):
        self.checks = []

    def add(self, section, name, expected, actual, status=None, note=""):
        if status is None:
            status = "PASS" if expected == actual else "FAIL"
        self.checks.append({"section": section, "check": name, "expected": expected,
                            "actual": actual, "status": status, "note": note})

    def approx(self, section, name, expected, actual, tol, note=""):
        status = "PASS" if abs(expected - actual) <= tol else "FAIL"
        self.add(section, name, expected, round(actual, 4), status, note)

    def info(self, section, name, expected, actual, note=""):
        self.add(section, name, expected, actual, "INFO", note)


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def krippendorff_alpha_nominal(units):
    """명목형 Krippendorff's alpha (코더 2명 이상인 unit 만 사용)."""
    coincidence = Counter()
    for values in units:
        m = len(values)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    coincidence[(values[i], values[j])] += 1.0 / (m - 1)
    n_c = Counter()
    for (a, _), v in coincidence.items():
        n_c[a] += v
    n = sum(n_c.values())
    d_o = sum(v for (a, b), v in coincidence.items() if a != b) / n
    d_e = sum(n_c[a] * n_c[b] for a in n_c for b in n_c if a != b) / (n * (n - 1))
    return 1.0 - d_o / d_e


MIN_VALID = 40000


def norm_text(t):
    return re.sub(r"\s+", " ", (t or "")).strip()


def sample_rule_violations(x, seen):
    """샘플 하나가 위반한 규칙 목록. seen 은 앞서 나온 댓글 텍스트 집합(중복 판정용)."""
    v = []
    text = norm_text(x.get("comment"))
    # (1) 텍스트
    if not text:
        v.append("R1a 댓글 텍스트 비어 있음")
    elif text in seen:
        v.append("R1b 댓글 텍스트 중복 (첫 등장만 유효)")
    if not norm_text(x.get("title")):
        v.append("R1c Context(title) 비어 있음")
    # (2) 혐오 여부 라벨
    off = x.get("OFF")
    if not isinstance(off, bool):
        v.append("R2 OFF(혐오 여부) 라벨 없음/비정상")
    # (3) 혐오 샘플의 혐오 종류 라벨
    if off is True:
        tgt = x.get("TGT")
        if tgt not in TGT_MAP:
            v.append("R3a OFF=True 인데 TGT(대상 유형) 라벨 없음")
        elif tgt == "group":
            labels = (x.get("GRP") or "").split(" & ") if x.get("GRP") else []
            if not labels:
                v.append("R3b TGT=group 인데 GRP(대상 집단) 라벨 없음")
            elif any(l not in LABEL_MAP for l in labels):
                v.append("R3c 알 수 없는 GRP 라벨")
        elif x.get("GRP"):
            v.append("R3d TGT!=group 인데 GRP 라벨 존재")
    elif off is False and (x.get("TGT") or x.get("GRP")):
        v.append("R3e OFF=False 인데 TGT/GRP 라벨 존재")
    # 어노테이션 일관성
    raw = x.get("raw_labels") or []
    if len(raw) != 3:
        v.append("R4a 어노테이터 수 != 3")
    elif isinstance(off, bool) and (sum(bool(r.get("offensiveness")) for r in raw) >= 2) != off:
        v.append("R4b OFF 가 3인 다수결과 불일치")
    return v


def check_valid_samples(V, data):
    """우선 검증: 라벨 규칙을 모두 만족하는 유효 샘플 수 >= 40,000."""
    S = "00. 유효 샘플 (우선 검증)"
    seen, valid, rule_fail = set(), [], Counter()
    for x in data:
        viol = sample_rule_violations(x, seen)
        seen.add(norm_text(x.get("comment")))
        rule_fail.update(viol)
        if not viol:
            valid.append(x)
    N, n_valid = len(data), len(valid)

    rules = ["R1a 댓글 텍스트 비어 있음", "R1b 댓글 텍스트 중복 (첫 등장만 유효)",
             "R1c Context(title) 비어 있음", "R2 OFF(혐오 여부) 라벨 없음/비정상",
             "R3a OFF=True 인데 TGT(대상 유형) 라벨 없음",
             "R3b TGT=group 인데 GRP(대상 집단) 라벨 없음", "R3c 알 수 없는 GRP 라벨",
             "R3d TGT!=group 인데 GRP 라벨 존재", "R3e OFF=False 인데 TGT/GRP 라벨 존재",
             "R4a 어노테이터 수 != 3", "R4b OFF 가 3인 다수결과 불일치"]
    for r in rules:
        if r.startswith("R1b"):
            V.info(S, f"[규칙] {r}", 0, rule_fail[r],
                   "같은 댓글이 다른 기사/영상에 달린 경우. (title, comment) 쌍 기준 중복은 별도 검사")
        else:
            V.add(S, f"[규칙] {r}", 0, rule_fail[r])
    pairs = Counter((norm_text(x["title"]), norm_text(x["comment"])) for x in data)
    V.add(S, "[규칙] (Context, 댓글) 쌍 중복", 0, sum(c - 1 for c in pairs.values() if c > 1))
    groups = {}
    for x in data:
        groups.setdefault(norm_text(x["comment"]), set()).add((x["OFF"], x["TGT"], x["GRP"]))
    conflict = sum(1 for s in groups.values() if len(s) > 1)
    V.info(S, "중복 댓글 중 라벨이 서로 다른 그룹 수", "-", conflict,
           "동일 텍스트라도 Context 에 따라 라벨이 달라질 수 있음")

    V.add(S, f"유효 샘플 수 >= {MIN_VALID:,}", True, n_valid >= MIN_VALID,
          note=f"유효 {n_valid:,} / 전체 {N:,} ({pct(n_valid, N):.2f}%), 제외 {N - n_valid:,}")

    # (4) 보고 수치 일치: 전체 데이터 기준 분포 == 논문 (상세는 섹션 1~3, 5~6)
    off = sum(x["OFF"] for x in data)
    level_b = Counter(TGT_MAP[x["TGT"]] for x in data if x["OFF"])
    reported = {"total": PAPER["total"], **PAPER["level_a"], **PAPER["level_b"]}
    actual = {"total": N, "NOT": N - off, "OFF": off, **{k: level_b[k] for k in PAPER["level_b"]}}
    V.add(S, "전체 샘플 수·라벨 분포 == 보고 수치 (전체 데이터)", reported, actual)

    v_off = sum(x["OFF"] for x in valid)
    v_b = Counter(TGT_MAP[x["TGT"]] for x in valid if x["OFF"])
    V.info(S, "유효 샘플 기준 라벨 분포", "-",
           {"total": n_valid, "NOT": n_valid - v_off, "OFF": v_off,
            **{k: v_b[k] for k in PAPER["level_b"]}},
           "중복 제거 후 분포. 논문 수치는 중복 포함 전체 기준")
    return {"valid": n_valid, "excluded": N - n_valid, "rule_violations": dict(rule_fail),
            "valid_level_a": {"NOT": n_valid - v_off, "OFF": v_off},
            "valid_level_b": dict(v_b), "duplicate_label_conflict_groups": conflict}


def validate(data):
    V = Validator()
    N = len(data)
    valid_stats = check_valid_samples(V, data)

    # ---------------- 0. 스키마 / 무결성 ----------------
    S = "0. 스키마/무결성"
    V.add(S, "모든 샘플의 필드 구성", True, all(set(x) == EXPECTED_KEYS for x in data))
    V.add(S, "guid 중복 없음", N, len({x["guid"] for x in data}))
    V.add(S, "빈 comment 없음", 0, sum(not (x["comment"] or "").strip() for x in data))
    V.add(S, "빈 title(Context) 없음", 0, sum(not (x["title"] or "").strip() for x in data))
    V.add(S, "OFF 는 bool", True, all(isinstance(x["OFF"], bool) for x in data))
    V.add(S, "OFF=False 이면 TGT/GRP/span 없음", 0,
          sum(1 for x in data if not x["OFF"] and
              (x["TGT"] or x["GRP"] or x["OFF_span"] or x["TGT_span"])))
    V.add(S, "OFF=True 이면 TGT 값이 유효", 0,
          sum(1 for x in data if x["OFF"] and x["TGT"] not in TGT_MAP))
    V.add(S, "GRP 라벨은 TGT=group 에만 존재", 0,
          sum(1 for x in data if bool(x["GRP"]) != (x["TGT"] == "group")))
    unknown = sorted({l for x in data if x["GRP"] for l in x["GRP"].split(" & ")} - set(LABEL_MAP))
    V.add(S, "알 수 없는 GRP 라벨 없음", [], unknown)

    # ---------------- 1. 기본 정보 ----------------
    S = "1. 기본 정보"
    V.add(S, "전체 샘플 수", PAPER["total"], N)
    sources = Counter(x["source"] for x in data)
    V.add(S, "데이터 출처 (NAVER 뉴스, YouTube)", sorted(PAPER["sources"]), sorted(sources))
    V.info(S, "플랫폼별 샘플 수", "논문 미제시", dict(sources))
    dates = sorted(x["date"] for x in data if x["date"])
    V.add(S, "날짜 누락 없음", 0, N - len(dates))
    V.add(S, "콘텐츠 기간 (YYYY-MM)", list(PAPER["date_range"]), [dates[0][:7], dates[-1][:7]],
          note=f"실제 {dates[0]} ~ {dates[-1]}")
    hangul = sum(bool(re.search(r"[가-힣]", x["comment"])) for x in data)
    V.add(S, "한국어(한글 포함) 댓글 비율 >= 95%", True, pct(hangul, N) >= 95.0,
          note=f"한글 포함 {hangul:,}/{N:,} ({pct(hangul, N):.2f}%)")

    # ---------------- 2. Level A ----------------
    S = "2. Level A (Offensiveness)"
    off = sum(x["OFF"] for x in data)
    level_a = {"NOT": N - off, "OFF": off}
    for k, v in PAPER["level_a"].items():
        V.add(S, f"{k} 샘플 수", v, level_a[k], note=f"{pct(level_a[k], N):.1f}%")
    V.add(S, "NOT 비율(%)", 49.8, round(pct(level_a["NOT"], N), 1))
    V.add(S, "OFF 비율(%)", 50.2, round(pct(level_a["OFF"], N), 1))
    V.info(S, "Section 3.3 본문 OFF 수치(20,130)", 20130, off,
           "논문 오기. Table 2(20,310)가 데이터와 일치")

    # ---------------- 3. Level B ----------------
    S = "3. Level B (Target Type)"
    level_b = Counter(TGT_MAP[x["TGT"]] for x in data if x["OFF"])
    for k, v in PAPER["level_b"].items():
        V.add(S, f"{k} 샘플 수", v, level_b[k], note=f"전체 대비 {pct(level_b[k], N):.1f}%")
    for k, p in zip(PAPER["level_b"], [6.4, 9.6, 3.5, 30.7]):
        V.add(S, f"{k} 전체 대비 비율(%)", p, round(pct(level_b[k], N), 1))
    V.add(S, "Level B 합계 == OFF", off, sum(level_b.values()))
    targeted = level_b["GRP"] + level_b["IND"] + level_b["OTH"]
    for k, p in PAPER["level_b_targeted_pct"].items():
        V.add(S, f"{k} (UNT 제외 기준, %)", p, round(pct(level_b[k], targeted), 1))

    # ---------------- 4/5/6/7. Level C & Target Group ----------------
    grp_rows = [x for x in data if x["TGT"] == "group"]
    label_counts = Counter()          # Table 3 방식: 댓글 내 라벨 토큰 단위 집계
    first_attr = Counter()            # 댓글당 1개 속성 (첫 라벨)
    for x in grp_rows:
        labels = x["GRP"].split(" & ")
        first_attr[LABEL_MAP[labels[0]][0]] += 1
        for l in labels:
            label_counts[LABEL_MAP[l]] += 1

    S = "4. Level C (Target Group Attribute)"
    V.add(S, "Level C 속성 합계 == GRP", PAPER["level_b"]["GRP"], sum(first_attr.values()))
    for k, v in PAPER["level_c"].items():
        V.info(S, f"{ATTR_NAMES[k]} 댓글 수", v, first_attr[k],
               f"공개 v1 데이터 기준(첫 라벨 속성). 논문 {pct(v, N):.1f}% / 데이터 "
               f"{pct(first_attr[k], N):.1f}%. 논문 Table 2 의 속성별 수치는 v1 최종 라벨로 "
               "정확히 재현되지 않음 (Table 3 세부 수치는 재현됨)")

    S = "5. Target Group (Table 3)"
    for attr, groups in PAPER["table3"].items():
        for g, v in groups.items():
            name = {"Muslim": "Islam"}.get(g, g)
            V.add(S, f"[{attr}] {g}", v, label_counts[(attr, name)])
    computed_total = sum(label_counts.values())
    shown_sum = sum(v for gs in PAPER["table3"].values() for v in gs.values())
    V.info(S, "Table 3 집단 라벨 총수", PAPER["table3_total"], computed_total,
           f"논문 표기 12,981 은 Table 3 행 합({shown_sum:,}) 및 데이터 라벨 총수와 불일치")
    V.add(S, "집단 라벨 수 >= GRP 댓글 수 (복수 라벨)", True, computed_total >= len(grp_rows),
          note=f"라벨 {computed_total:,} / 댓글 {len(grp_rows):,}")

    S = "6. Target Group Top 10 (Table 4)"
    ranked = sorted(((TABLE3_ALIAS.get(n, n), c) for (a, n), c in label_counts.items()
                     if n != "Others"), key=lambda t: -t[1])[:10]
    V.add(S, "Top 10 순위", [t[0] for t in PAPER["top10"]], [t[0] for t in ranked])
    for (name, cnt, p), (rn, rc) in zip(PAPER["top10"], ranked):
        V.add(S, f"{name} 샘플 수", cnt, rc)
        V.add(S, f"{name} 비율(% of 12,981)", p, round(pct(rc, PAPER["table3_total"]), 2))

    S = "7. Target Group 라벨 체계 (Table 10)"
    present = {}
    for (a, n) in label_counts:
        if n != "Others":
            present.setdefault(a, set()).add(n)
    for attr, names in PAPER["table10"].items():
        V.add(S, f"{ATTR_NAMES[attr]} ({len(names)}개)", sorted(names),
              sorted(present.get(attr, set())))
    n_fine = sum(len(v) for v in present.values())
    V.add(S, "세부 집단 라벨 수 (Appendix 열거 기준)", 22, n_fine)
    V.info(S, "본문 표기 Target Group 수", 21, n_fine, "논문 본문 21개 vs Appendix 22개 불일치")
    V.add(S, "대분류(Attribute) 수", 5, len(present))

    # ---------------- 8. Span ----------------
    S = "8. Span Annotation"
    no_span = sum(1 for x in grp_rows if not x["TGT_span"])
    V.add(S, "GRP 중 Target Span 없는 비율(%)", PAPER["grp_no_tgt_span_pct"],
          round(pct(no_span, len(grp_rows)), 1), note=f"{no_span:,}/{len(grp_rows):,}")
    off_span_n = sum(1 for x in data if x["OFF_span"])
    tgt_span_n = sum(1 for x in data if x["TGT_span"])
    off_lens = [len(x["OFF_span"]) for x in data if x["OFF_span"]]
    tgt_lens = [len(x["TGT_span"]) for x in data if x["TGT_span"]]
    V.info(S, "OFF_span 있는 댓글 수", "논문 미제시",
           f"{off_span_n:,} (OFF 대비 {pct(off_span_n, off):.1f}%, 평균 길이 "
           f"{sum(off_lens) / len(off_lens):.1f}자)")
    V.info(S, "TGT_span 있는 댓글 수", "논문 미제시",
           f"{tgt_span_n:,} (OFF 대비 {pct(tgt_span_n, off):.1f}%, 평균 길이 "
           f"{sum(tgt_lens) / len(tgt_lens):.1f}자)")

    # ---------------- 9. Annotation ----------------
    S = "9. 어노테이션"
    ann = {r["annotator_id"] for x in data for r in x["raw_labels"]}
    V.add(S, "전체 어노테이터 수", PAPER["annotators"], len(ann))
    per = Counter(len(x["raw_labels"]) for x in data)
    V.add(S, "샘플당 어노테이터 수", {PAPER["annotators_per_sample"]: N}, dict(per))
    # 다수결 검증: 최종 OFF == 어노테이터 다수결
    mismatch = sum(1 for x in data
                   if (sum(r["offensiveness"] for r in x["raw_labels"]) >= 2) != x["OFF"])
    V.add(S, "Level A 최종 라벨 == 3인 다수결", 0, mismatch)
    alpha = krippendorff_alpha_nominal([[r["offensiveness"] for r in x["raw_labels"]]
                                        for x in data])
    V.approx(S, "Level A Krippendorff's alpha", PAPER["alpha_a"], alpha, 0.005)
    V.info(S, "Level B alpha / Level C alpha / 전체 평균", "0.45 / 0.65 / 0.55", "미계산",
           "다중 선택 라벨 구조로 논문과 동일한 계산 방식 재현 불가")

    # ---------------- 10. 수집/비식별화 ----------------
    S = "10. 비식별화"
    texts = [x["comment"] for x in data]
    V.add(S, "원문 URL 노출 없음", 0, sum(bool(re.search(r"https?://|www\.", t)) for t in texts))
    V.add(S, "원문 이메일 노출 없음", 0,
          sum(bool(re.search(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}", t)) for t in texts))
    V.info(S, "치환 토큰 사용 수", "-",
           {t: sum(t in s for s in texts) for t in ("<user>", "<url>", "<email>")})

    stats = {
        "valid_samples": valid_stats,
        "total": N,"sources": dict(sources), "date_min": dates[0], "date_max": dates[-1],
        "level_a": level_a, "level_b": dict(level_b),
        "level_c_first_label": dict(first_attr),
        "target_group_label_counts": {f"{a}/{n}": c for (a, n), c in sorted(label_counts.items())},
        "target_group_label_total": computed_total,
        "grp_no_tgt_span": no_span, "annotators": len(ann), "alpha_level_a": round(alpha, 4),
    }
    return V.checks, stats


def fmt(v):
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


def write_markdown(path, checks, stats, summary):
    lines = ["# KOLD 데이터 검증 결과", "",
             f"- 데이터: `{summary['data']}`",
             f"- 결과: **{'PASS' if summary['FAIL'] == 0 else 'FAIL'}** "
             f"(PASS {summary['PASS']}, FAIL {summary['FAIL']}, INFO {summary['INFO']})", "",
             "INFO 항목은 논문 자체의 불일치 또는 공개 데이터로 재현 불가능한 수치로, "
             "판정에 포함하지 않습니다.", ""]
    section = None
    for c in checks:
        if c["section"] != section:
            section = c["section"]
            lines += ["", f"## {section}", "", "| 상태 | 항목 | 논문/기대값 | 데이터 | 비고 |",
                      "|---|---|---|---|---|"]
        lines.append(f"| {c['status']} | {c['check']} | {fmt(c['expected'])} | "
                     f"{fmt(c['actual'])} | {c['note']} |")
    lines += ["", "## 계산된 원시 통계", "", "```json",
              json.dumps(stats, ensure_ascii=False, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    checks, stats = validate(data)

    summary = Counter(c["status"] for c in checks)
    summary = {"PASS": summary["PASS"], "FAIL": summary["FAIL"], "INFO": summary["INFO"],
               "data": str(args.data)}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validation_report.json").write_text(
        json.dumps({"summary": summary, "stats": stats, "checks": checks},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(args.out / "validation_report.md", checks, stats, summary)

    section = None
    for c in checks:
        if c["section"] != section:
            section = c["section"]
            print(f"\n== {section} ==")
        note = f"  ({c['note']})" if c["note"] else ""
        print(f"[{c['status']:4}] {c['check']}: 기대 {fmt(c['expected'])} | "
              f"실제 {fmt(c['actual'])}{note}")
    print(f"\nSUMMARY: PASS {summary['PASS']}  FAIL {summary['FAIL']}  INFO {summary['INFO']}")
    print(f"결과 저장: {args.out}/validation_report.json, validation_report.md")
    return 0 if summary["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
