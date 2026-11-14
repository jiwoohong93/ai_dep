#!/usr/bin/env python3
"""KOLD 혐오 표현 분석 데모.

댓글(+선택: 기사/영상 제목)을 넣으면 논문 구조대로 단계별 예측을 보여준다.
    Level A  offensive 여부
    Level B  target type (UNT / IND / OTH / GRP)          -- OFF 일 때
    Level C  target group (27개 세부 그룹)                  -- GRP 일 때
    span     offensive span, target span (댓글 안 위치)     -- OFF 일 때

모델은 sweep 의 final 단계에서 저장된 것(runs/final/<task>/<model>/s1/model)을 쓴다.
task 마다 results.json 의 test F1 이 가장 높은 모델을 고르고, 없으면 저장된 아무 모델이나 쓴다.

실행:
    python language_2/demo.py --text "댓글" [--title "제목"]   # 한 문장
    python language_2/demo.py                                   # 대화형 (터미널)
    python language_2/demo.py --web [--port 7860] [--share]     # 웹 UI (gradio 필요)
    python language_2/demo.py --models_dir DIR                  # DIR/<task>/model 구조의 모델 사용
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import MAX_LEN, decode_span  # noqa: E402

TASKS = ["A", "B", "C", "off_span", "tgt_span"]
MODEL_PREF = ["roberta-large", "roberta-base", "bert-base"]
B_NAMES = {"UNT": "특정 대상 없음 (untargeted)", "IND": "개인 (individual)",
           "OTH": "기타 (other)", "GRP": "집단 (group)"}


def find_models(models_dir=None):
    """task -> model 디렉토리."""
    found = {}
    if models_dir:
        for t in TASKS:
            for cand in (Path(models_dir) / t / "model", Path(models_dir) / t):
                if (cand / "config.json").exists():
                    found[t] = cand
                    break
        return found
    final = HERE / "runs" / "final"
    results = {}
    if (HERE / "runs" / "results.json").exists():
        results = json.load(open(HERE / "runs" / "results.json"))
    for t in TASKS:
        cands = [m for m in MODEL_PREF if (final / t / m / "s1" / "model" / "config.json").exists()]
        if not cands:
            continue
        best = max(cands, key=lambda m: results.get(f"{t}/{m}", {}).get("test", {}).get("f1", -MODEL_PREF.index(m)))
        found[t] = final / t / best / "s1" / "model"
    return found


class KOLDPredictor:
    def __init__(self, model_dirs, device=None):
        missing = [t for t in TASKS if t not in model_dirs]
        if missing:
            raise SystemExit(f"모델이 없는 task: {missing}. sweep 이 끝났는지, 또는 --models_dir 를 확인하세요.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.paths, self.tok, self.models = model_dirs, {}, {}
        for t, p in model_dirs.items():
            cls = AutoModelForTokenClassification if t.endswith("_span") else AutoModelForSequenceClassification
            self.tok[t] = AutoTokenizer.from_pretrained(p)
            self.models[t] = cls.from_pretrained(p).to(self.device).eval()

    @torch.no_grad()
    def _classify(self, t, title, comment):
        enc = self.tok[t](title, comment, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(self.device)
        probs = self.models[t](**enc).logits.softmax(-1)[0].tolist()
        id2label = self.models[t].config.id2label
        return sorted(((id2label[i], p) for i, p in enumerate(probs)), key=lambda x: -x[1])

    @torch.no_grad()
    def _span(self, t, title, comment):
        tok = self.tok[t]
        enc = tok(title, comment, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        tokens = [(i, s, e) for i, ((s, e), sid) in enumerate(zip(enc["offset_mapping"], enc.sequence_ids()))
                  if sid == 1 and s != e]
        inputs = {k: torch.tensor([v]).to(self.device) for k, v in enc.items() if k != "offset_mapping"}
        pred = self.models[t](**inputs).logits.argmax(-1)[0].tolist()
        chars = sorted(decode_span(pred, tokens))
        spans, start = [], None  # 문자 집합 -> [start, end) 구간
        for k, c in enumerate(chars):
            if start is None:
                start = c
            if k + 1 == len(chars) or chars[k + 1] != c + 1:
                spans.append((start, c + 1))
                start = None
        return [(s, e, comment[s:e]) for s, e in spans]

    def predict(self, comment, title=""):
        title = title or ""
        out = {"title": title, "comment": comment}
        a = self._classify("A", title, comment)
        p_off = dict(a)["OFF"]
        out["offensive"] = {"label": "OFF" if p_off >= 0.5 else "NOT", "prob_off": p_off}
        if p_off < 0.5:
            return out
        b = self._classify("B", title, comment)
        out["target_type"] = {"label": b[0][0], "probs": dict(b)}
        out["offensive_span"] = self._span("off_span", title, comment)
        if b[0][0] != "UNT":
            out["target_span"] = self._span("tgt_span", title, comment)
        if b[0][0] == "GRP":
            out["target_group"] = {"label": self._classify("C", title, comment)[0][0],
                                   "top3": self._classify("C", title, comment)[:3]}
        return out


def format_text(r):
    lines = [f"댓글: {r['comment']}" + (f"   (제목: {r['title']})" if r["title"] else "")]
    o = r["offensive"]
    lines.append(f"[A] {'공격적 (OFF)' if o['label'] == 'OFF' else '공격적이지 않음 (NOT)'}  P(OFF)={o['prob_off']:.2f}")
    if o["label"] == "OFF":
        tt = r["target_type"]
        lines.append(f"[B] 대상 유형: {B_NAMES[tt['label']]}  ({tt['probs'][tt['label']]:.2f})")
        if "target_group" in r:
            top = ", ".join(f"{l} {p:.2f}" for l, p in r["target_group"]["top3"])
            lines.append(f"[C] 대상 집단: {r['target_group']['label']}   (top3: {top})")
        lines.append("[span] 공격 표현: " + (", ".join(f"'{s}'" for _, _, s in r["offensive_span"]) or "(없음)"))
        if "target_span" in r:
            lines.append("[span] 대상 표현: " + (", ".join(f"'{s}'" for _, _, s in r["target_span"]) or "(없음 / 명시되지 않음)"))
    return "\n".join(lines)


def highlight(r):
    """gradio HighlightedText 용 [(text, label|None), ...]."""
    c = r["comment"]
    lab = [None] * len(c)
    for s, e, _ in r.get("target_span", []):
        for k in range(s, e):
            lab[k] = "대상"
    for s, e, _ in r.get("offensive_span", []):
        for k in range(s, e):
            lab[k] = "공격+대상" if lab[k] == "대상" else "공격"
    out = []
    for ch, l in zip(c, lab):
        if out and out[-1][1] == l:
            out[-1][0] += ch
        else:
            out.append([ch, l])
    return [tuple(x) for x in out]


def launch_web(pred, port, share):
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("gradio 가 설치되어 있지 않습니다:  pip install gradio==5.49.1  (CLI 모드는 --web 없이 실행)")

    def run(title, comment):
        if not comment.strip():
            return "댓글을 입력하세요.", [], {}
        r = pred.predict(comment.strip(), title.strip())
        return format_text(r), highlight(r), r

    with gr.Blocks(title="KOLD 혐오 표현 분석") as ui:
        gr.Markdown("## KOLD 혐오 표현 분석 데모\n댓글을 넣으면 공격성 → 대상 유형 → 대상 집단 → 표현 위치 순서로 예측합니다. "
                    "제목(기사/영상)은 선택이지만, 넣으면 대상 집단 예측이 더 정확해집니다.")
        with gr.Row():
            with gr.Column():
                title = gr.Textbox(label="제목 (선택)", placeholder="기사 또는 영상 제목")
                comment = gr.Textbox(label="댓글", lines=4)
                btn = gr.Button("분석", variant="primary")
            with gr.Column():
                summary = gr.Textbox(label="예측 결과", lines=7)
                hl = gr.HighlightedText(label="표현 위치", color_map={"공격": "red", "대상": "blue", "공격+대상": "purple"})
                raw = gr.JSON(label="원본 출력")
        btn.click(run, [title, comment], [summary, hl, raw])
        comment.submit(run, [title, comment], [summary, hl, raw])
        gr.Markdown("모델: " + ", ".join(f"{t}={Path(p).parent.parent.name if 'final' in str(p) else p}"
                                        for t, p in pred.paths.items()))
    ui.launch(server_name="0.0.0.0", server_port=port, share=share)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--title", default="")
    ap.add_argument("--models_dir")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="gradio 공개 링크 생성 (외부 공유)")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    args = ap.parse_args()

    dirs = find_models(args.models_dir)
    pred = KOLDPredictor(dirs)
    print("모델:", {t: str(p) for t, p in dirs.items()}, file=sys.stderr)

    if args.web:
        launch_web(pred, args.port, args.share)
    elif args.text:
        r = pred.predict(args.text, args.title)
        print(json.dumps(r, ensure_ascii=False, indent=2) if args.json else format_text(r))
    else:
        print("댓글을 입력하세요 (제목을 함께 넣으려면 '제목 || 댓글', 종료는 빈 줄).")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                break
            title, comment = line.split("||", 1) if "||" in line else ("", line)
            print(format_text(pred.predict(comment.strip(), title.strip())), "\n")


if __name__ == "__main__":
    main()
