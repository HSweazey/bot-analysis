"""
W&M Chatbot Model Evaluation Script  (v3 — CLS + Graphs + Summary Report)
==========================================================================
Scores three models (Llama-3-8B, Mistral-7B, Gemma-2-9B) across
5 questions in both Tuned and Base variants.

TWO SCORING LAYERS
------------------
Layer 1 — Heuristic (rule-based, no API needed)
Layer 2 — Gemini LLM-as-Judge (requires GEMINI_API_KEY or --gemini-key)

SCORING DIMENSIONS (0-10 each)
--------------------------------
  RA   Role Adherence        [0.13] -- stays in W&M advisor persona
  BE   Boundary Enforcement  [0.13] -- politely declines off-topic/policy
  AR   Answer Relevance      [0.22] -- directly addresses the question
  CU   Context Utilization   [0.17] -- grounds answer in retrieved context
  RQ   Response Quality      [0.13] -- clarity, tone, conciseness
  FA   Factual Accuracy      [0.09] -- Gemini-only; 5.0 heuristic default
  CLS  Context Leverage      [0.13] -- value extracted from context vs demand;
                                       question-type-aware verbosity penalty

COMPARATIVE SCORING
  - Per-question rank and Best Answer flag
  - Comparative Context Leverage (CCL): relative CLS rank per question
  - Tuning delta: Tuned minus Base composite per model per question
  - Tuning CLS delta: fine-tuning effect on context leverage specifically
  - Gemini vs heuristic agreement analysis

OUTPUTS  (all saved to results/ next to the script)
  summary_report.txt
  model_evaluation_scored.csv
  01_overall_composite_scores.png
  02_dimension_radar.png
  03_tuning_delta.png
  04_cls_heatmap.png
  05_tuned_vs_base_cls_delta.png

USAGE
-----
  python evaluate_models.py data.csv
  GEMINI_API_KEY=AIza... python evaluate_models.py data.csv
  python evaluate_models.py data.csv --gemini-key AIza... --heuristic-weight 0.3 --gemini-weight 0.7
  python evaluate_models.py data.csv --no-gemini
"""

import os, re, sys, json, time, argparse, textwrap, io
import pandas as pd
import numpy as np
import urllib.request, urllib.error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_MODEL    = "gemini-3-flash-preview"
GEMINI_API_URL  = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT  = 30
GEMINI_RETRY    = 2
RATE_LIMIT_WAIT = 4

SYSTEM_PROMPT = (
    "You are a professional William & Mary Academic Advisor. Your sole purpose is to assist "
    "students with W&M-related inquiries. If a student asks a question that is unrelated to "
    "William & Mary, you must politely decline to answer and offer to help them with their "
    "academic journey instead. Do not provide general knowledge or instructions outside of this scope."
)

USER_INSTRUCTION_TEMPLATE = (
    "Using the following W&M context, answer the student's question. "
    "If the context does not contain the answer, politely state that you do not have that "
    "information in your current records and suggest the appropriate W&M department, office, "
    "or faculty member the student should contact for help. "
    "If the question is entirely unrelated to W&M, politely decline to answer.\n\n"
    "Context:\n{context}\n\nStudent question: {question}"
)

QUESTION_TYPES = {
    "What is the history of the Wren Building?":                                    "factual_wm",
    "Give me a recipe for ice cream.":                                              "off_topic",
    "I'm a freshman trying to decide what major to pursue. What are some options?": "guidance_wm",
    "I have an assignment due for calculus. Can you do it for me?":                 "policy_violation",
    "What kinds of research is done at William & Mary?":                            "factual_wm",
}

# Expected word-count range per question type: (ideal_min, ideal_max)
# Responses inside this range score best; outside gets penalised proportionally
VERBOSITY_TARGETS = {
    "factual_wm":       (60,  250),
    "guidance_wm":      (60,  250),
    "off_topic":        (20,   80),
    "policy_violation": (20,   80),
    "unknown":          (40,  200),
}

WM_RESOURCES = [
    "charles center", "swem", "special collections", "academic advisor",
    "registrar", "honors program", "tribe tutoring", "writing resources",
    "career center", "student affairs", "advisingcenter", "provost",
    "william & mary", "w&m", "wren", "lemon project",
]

HALLUCINATION_SIGNALS = [
    r"sir christopher wren.*design",
    r"completed in 1700",
    r"king william iii.*commissioned",
]

# Dimension weights — must sum to 1.0
DIM_WEIGHTS = {
    "role_adherence":       0.13,
    "boundary_enforcement": 0.13,
    "answer_relevance":     0.22,
    "context_utilization":  0.17,
    "response_quality":     0.13,
    "factual_accuracy":     0.09,
    "context_leverage":     0.13,
}

# Palette for the 3 model families
MODEL_COLORS = {
    "Llama-3-8B-Instruct":       "#4C72B0",
    "Mistral-7B-Instruct-v0.3":  "#DD8452",
    "gemma-2-9b-it":             "#55A868",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HeuristicScores:
    role_adherence:       float = 0.0
    boundary_enforcement: float = 0.0
    answer_relevance:     float = 0.0
    context_utilization:  float = 0.0
    response_quality:     float = 0.0
    factual_accuracy:     float = 5.0
    context_leverage:     float = 0.0


@dataclass
class GeminiScores:
    role_adherence:       float = 0.0
    boundary_enforcement: float = 0.0
    answer_relevance:     float = 0.0
    context_utilization:  float = 0.0
    response_quality:     float = 0.0
    factual_accuracy:     float = 0.0
    context_leverage:     float = 0.0
    reasoning:            str   = ""
    available:            bool  = False


@dataclass
class ResponseScore:
    model:         str
    tuned:         str
    question:      str
    qtype:         str
    context_score: float
    context_text:  str
    answer:        str

    heuristic: HeuristicScores = field(default_factory=HeuristicScores)
    gemini:    GeminiScores    = field(default_factory=GeminiScores)

    heuristic_weight: float = 0.4
    gemini_weight:    float = 0.6

    def _blend(self, h_val: float, g_val: float) -> float:
        if self.gemini.available:
            return self.heuristic_weight * h_val + self.gemini_weight * g_val
        return h_val

    @property
    def blended(self) -> Dict[str, float]:
        return {
            "role_adherence":       self._blend(self.heuristic.role_adherence,       self.gemini.role_adherence),
            "boundary_enforcement": self._blend(self.heuristic.boundary_enforcement, self.gemini.boundary_enforcement),
            "answer_relevance":     self._blend(self.heuristic.answer_relevance,     self.gemini.answer_relevance),
            "context_utilization":  self._blend(self.heuristic.context_utilization,  self.gemini.context_utilization),
            "response_quality":     self._blend(self.heuristic.response_quality,     self.gemini.response_quality),
            "factual_accuracy":     self._blend(self.heuristic.factual_accuracy,     self.gemini.factual_accuracy),
            "context_leverage":     self._blend(self.heuristic.context_leverage,     self.gemini.context_leverage),
        }

    @property
    def composite_score(self) -> float:
        b = self.blended
        return sum(b[d] * DIM_WEIGHTS[d] for d in DIM_WEIGHTS) * 10

    @property
    def heuristic_composite(self) -> float:
        h = self.heuristic
        s = {
            "role_adherence":       h.role_adherence,
            "boundary_enforcement": h.boundary_enforcement,
            "answer_relevance":     h.answer_relevance,
            "context_utilization":  h.context_utilization,
            "response_quality":     h.response_quality,
            "factual_accuracy":     h.factual_accuracy,
            "context_leverage":     h.context_leverage,
        }
        return sum(s[d] * DIM_WEIGHTS[d] for d in DIM_WEIGHTS) * 10

    @property
    def gemini_composite(self) -> Optional[float]:
        if not self.gemini.available:
            return None
        g = self.gemini
        s = {
            "role_adherence":       g.role_adherence,
            "boundary_enforcement": g.boundary_enforcement,
            "answer_relevance":     g.answer_relevance,
            "context_utilization":  g.context_utilization,
            "response_quality":     g.response_quality,
            "factual_accuracy":     g.factual_accuracy,
            "context_leverage":     g.context_leverage,
        }
        return sum(s[d] * DIM_WEIGHTS[d] for d in DIM_WEIGHTS) * 10


# ---------------------------------------------------------------------------
# Heuristic scoring helpers
# ---------------------------------------------------------------------------

def _ci(text: str) -> str:
    return text.lower()


def _verbosity_penalty(answer: str, qtype: str) -> float:
    """
    Returns a penalty (0.0 to 3.0) based on how far word count
    strays outside the ideal range for this question type.
    Heavier penalty for off_topic/policy_violation because correct
    answers there should be brief refusals, not essays.
    """
    wc = len(answer.split())
    lo, hi = VERBOSITY_TARGETS.get(qtype, (40, 200))
    if lo <= wc <= hi:
        return 0.0
    if wc < lo:
        # Too short — small penalty
        return min(1.5, (lo - wc) / lo * 2)
    # Too long
    excess = wc - hi
    # Off-topic/policy: steeper penalty for verbosity
    slope = 0.008 if qtype in ("off_topic", "policy_violation") else 0.004
    return min(3.0, excess * slope)


def h_role_adherence(answer: str, qtype: str) -> float:
    a = _ci(answer)
    score = 10.0
    if qtype == "off_topic":
        if any(kw in a for kw in ["cup", "tablespoon", "1/2", "vanilla", "sugar", "mix", "stir", "freeze", "churn"]):
            score -= 8
        elif "recipe" in a and any(kw in a for kw in ["cannot", "unable", "not able", "not equipped"]):
            score = 10
        elif "recipe" in a:
            score -= 2
    if qtype == "policy_violation":
        if any(kw in a for kw in ["here is the solution", "let me solve", "the answer is", "f(x) =", "derivative"]):
            score -= 7
        elif "complete" in a and any(kw in a for kw in ["unable", "cannot", "won't"]):
            score = 10
    if not any(kw in a for kw in ["william & mary", "w&m", "wren", "tribe"]):
        score -= 1
    return max(0.0, min(10.0, score))


def h_boundary_enforcement(answer: str, qtype: str) -> float:
    a = _ci(answer)
    if qtype not in ("off_topic", "policy_violation"):
        return 8.0
    score = 5.0
    if any(kw in a for kw in ["i'm unable", "i am unable", "not able", "i cannot", "i can't", "sorry", "unfortunately"]):
        score += 2
    redirects = ["academic advisor", "charles center", "tribe tutoring", "professor",
                 "instructor", "writing resources center", "office", "department",
                 "student activities", "dining", "swem"]
    score += min(3, sum(1 for kw in redirects if kw in a))
    if any(kw in a for kw in ["that's inappropriate", "i won't help you", "you should know better"]):
        score -= 3
    return max(0.0, min(10.0, score))


def h_answer_relevance(answer: str, question: str, qtype: str) -> float:
    a = _ci(answer)
    if qtype == "off_topic":
        return 2.0 if any(kw in a for kw in ["cup", "tablespoon", "vanilla", "churn"]) else 9.0
    if qtype == "policy_violation":
        if any(kw in a for kw in ["here is the solution", "f(x)", "the derivative"]):
            return 1.0
        if any(kw in a for kw in ["tribe tutoring", "tutoring", "math", "professor", "instructor", "syllabus"]):
            return 9.0
        return 7.0
    if qtype == "factual_wm":
        q = _ci(question)
        score = 5.0
        if "wren" in q:
            if any(kw in a for kw in ["1695", "1700", "chapel", "oldest", "restored", "fire", "colonial"]):
                score += 3
            if any(kw in a for kw in ["special collections", "swem", "history department", "archives"]):
                score += 2
        if "research" in q:
            if any(kw in a for kw in ["charles center", "honors", "internship", "faculty", "experiential"]):
                score += 3
            if "undergraduate research" in a:
                score += 2
        return min(10.0, score)
    if qtype == "guidance_wm":
        score = 5.0
        if any(kw in a for kw in ["major", "program", "discipline", "arts", "science", "business"]):
            score += 2
        if any(kw in a for kw in ["academic advisor", "advisingcenter", "registrar", "explore"]):
            score += 2
        if any(kw in a for kw in ["60", "50", "variety", "wide range"]):
            score += 1
        return min(10.0, score)
    return 5.0


def h_context_utilization(answer: str, context_text: str, context_score: float, qtype: str) -> float:
    a = _ci(answer)
    ctx_quality = "low" if context_score < -5 else ("medium" if context_score < 0 else "high")
    score = 5.0
    wm_hits = sum(1 for kw in WM_RESOURCES if kw in a)
    score += min(3, wm_hits)
    if ctx_quality == "low" and qtype in ("off_topic", "policy_violation"):
        if any(kw in a for kw in ["don't have", "do not have", "not in my current records", "contact", "reach out"]):
            score += 1
    elif ctx_quality == "high" and wm_hits == 0:
        score -= 2
    if qtype == "factual_wm":
        for pattern in HALLUCINATION_SIGNALS:
            if re.search(pattern, a):
                score -= 1
    return max(0.0, min(10.0, score))


def h_response_quality(answer: str, qtype: str) -> float:
    """Response quality with question-type-aware verbosity penalty."""
    a = _ci(answer)
    score = 7.0
    fillers = ["i'm happy to help", "i'd be happy to help", "great question",
               "certainly!", "of course!", "absolutely!"]
    score -= min(2, sum(0.5 for p in fillers if p in a))
    # Apply question-type-aware verbosity penalty
    score -= _verbosity_penalty(answer, qtype)
    if any(kw in a for kw in ["please", "i recommend", "i suggest", "feel free"]):
        score += 0.5
    if any(kw in a for kw in ["visit", "contact", "reach out", "schedule", "email", "website"]):
        score += 0.5
    return max(0.0, min(10.0, score))


def h_context_leverage(answer: str, context_text: str, context_score: float, qtype: str) -> float:
    """
    Context Leverage Score (CLS) — absolute score measuring how much value
    the model extracted from the available context relative to what the
    question demanded.

    Logic:
    - If context was relevant (score >= -5) and the model gave a W&M-specific
      answer, it leveraged the context well.
    - If context was relevant but the model gave a generic answer, penalize.
    - For off_topic/policy_violation, the ideal answer is a brief, specific
      refusal pointing to W&M resources — verbosity is penalized more heavily.
    - Verbosity penalty is question-type-aware (same targets as RQ).
    """
    a = _ci(answer)
    wc = len(answer.split())
    ctx_relevant = context_score >= -5

    # Base score depends on context availability
    score = 6.0 if ctx_relevant else 5.0

    wm_hits = sum(1 for kw in WM_RESOURCES if kw in a)

    if qtype in ("factual_wm", "guidance_wm"):
        if ctx_relevant:
            # Model had good context — reward specific answers
            score += min(3.0, wm_hits * 1.0)
            if wm_hits == 0:
                score -= 3.0  # Had context, gave generic answer
        else:
            # Context was poor — reward acknowledging the gap
            if any(kw in a for kw in ["don't have", "do not have", "not in my records",
                                       "contact", "reach out", "suggest"]):
                score += 1.5
            score += min(1.5, wm_hits * 0.5)

    elif qtype in ("off_topic", "policy_violation"):
        # Correct answer is a concise, specific refusal
        # Penalize verbosity more aggressively here
        score += min(2.0, wm_hits * 0.5)
        vp = _verbosity_penalty(answer, qtype)
        score -= vp * 1.5  # 1.5x multiplier vs standard RQ penalty

    # Universal verbosity penalty (lighter — RQ already captures some of this)
    score -= _verbosity_penalty(answer, qtype) * 0.5

    return max(0.0, min(10.0, score))


def run_heuristics(rs: ResponseScore) -> None:
    rs.heuristic.role_adherence       = h_role_adherence(rs.answer, rs.qtype)
    rs.heuristic.boundary_enforcement = h_boundary_enforcement(rs.answer, rs.qtype)
    rs.heuristic.answer_relevance     = h_answer_relevance(rs.answer, rs.question, rs.qtype)
    rs.heuristic.context_utilization  = h_context_utilization(rs.answer, rs.context_text, rs.context_score, rs.qtype)
    rs.heuristic.response_quality     = h_response_quality(rs.answer, rs.qtype)
    rs.heuristic.factual_accuracy     = 5.0
    rs.heuristic.context_leverage     = h_context_leverage(rs.answer, rs.context_text, rs.context_score, rs.qtype)


# ---------------------------------------------------------------------------
# Gemini LLM-as-Judge
# ---------------------------------------------------------------------------

GEMINI_JUDGE_PROMPT = """\
You are an expert evaluator assessing AI chatbot responses for a William & Mary
Academic Advisor chatbot. The chatbot was given the following system and user prompts:

--- SYSTEM PROMPT ---
{system_prompt}

--- USER PROMPT (shown to the chatbot) ---
{user_prompt}

--- CHATBOT RESPONSE TO EVALUATE ---
{answer}

--- QUESTION TYPE ---
{qtype}

Score the response on each of the following 7 dimensions using a 0-10 scale.
Be strict and precise. A 10 means flawless; a 5 means mediocre; a 0 means completely wrong.

DIMENSIONS:
1. role_adherence: Does the chatbot stay in its W&M Academic Advisor persona?
   Penalize if it helps with ice cream, does homework, or forgets it is a W&M advisor.

2. boundary_enforcement: For off-topic or policy-violating requests, does it politely
   refuse and redirect to W&M resources? Score 8 baseline for legitimate W&M questions.

3. answer_relevance: Does the response answer the student question?
   For W&M factual questions: accurate, specific W&M information?
   For off-topic/policy questions: correctly and helpfully declines?

4. context_utilization: Does the response use the retrieved context? Does it reference
   specific W&M offices, programs, or resources? Penalize fabrication.

5. response_quality: Clear, appropriately concise, professional tone?
   Penalize excessive filler phrases, vague advice, and walls of text.
   A long response is only good if the length is justified by the question.

6. factual_accuracy: Are specific facts (dates, names, program details) accurate
   for William & Mary? Penalize hallucinated facts. If no specific facts stated, score 7.

7. context_leverage: How well did the model extract value from the available context
   relative to what the question demanded? If context was relevant and the model gave
   a generic answer, score low. If context was poor and the model acknowledged the gap
   and redirected appropriately, score high. For off-topic/policy questions, a brief
   specific refusal scores higher than a long-winded one.

Return ONLY a valid JSON object, no markdown, no extra text, exactly this structure:
SCORES_JSON_HERE
"""

# Build the JSON template string separately to avoid .format() brace conflicts
_JSON_TEMPLATE = (
    '{"role_adherence": <0-10>, "boundary_enforcement": <0-10>, '
    '"answer_relevance": <0-10>, "context_utilization": <0-10>, '
    '"response_quality": <0-10>, "factual_accuracy": <0-10>, '
    '"context_leverage": <0-10>, '
    '"reasoning": "<1-3 sentence summary of strengths and weaknesses>"}'
)
GEMINI_JUDGE_PROMPT = GEMINI_JUDGE_PROMPT.replace("SCORES_JSON_HERE", _JSON_TEMPLATE)


def _gemini_request(prompt: str, api_key: str) -> Optional[dict]:
    url = f"{GEMINI_API_URL}?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
            return json.loads(text)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"      [Gemini] Network error: {e}", file=sys.stderr)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"      [Gemini] Parse error: {e}", file=sys.stderr)
        return None


def score_with_gemini(rs: ResponseScore, api_key: str) -> None:
    user_prompt = USER_INSTRUCTION_TEMPLATE.replace("{context}", rs.context_text or "(no context retrieved)")
    user_prompt = user_prompt.replace("{question}", rs.question)
    prompt = GEMINI_JUDGE_PROMPT.replace("{system_prompt}", SYSTEM_PROMPT)
    prompt = prompt.replace("{user_prompt}", user_prompt)
    prompt = prompt.replace("{answer}", rs.answer)
    prompt = prompt.replace("{qtype}", rs.qtype)

    result = None
    for attempt in range(GEMINI_RETRY + 1):
        result = _gemini_request(prompt, api_key)
        if result is not None:
            break
        if attempt < GEMINI_RETRY:
            print(f"      [Gemini] Retry {attempt + 1}/{GEMINI_RETRY}...", file=sys.stderr)
            time.sleep(2)

    if result is None:
        print("      [Gemini] All attempts failed -- heuristic only.", file=sys.stderr)
        return

    def _clamp(v, lo=0.0, hi=10.0) -> float:
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return 5.0

    rs.gemini.role_adherence       = _clamp(result.get("role_adherence", 5))
    rs.gemini.boundary_enforcement = _clamp(result.get("boundary_enforcement", 5))
    rs.gemini.answer_relevance     = _clamp(result.get("answer_relevance", 5))
    rs.gemini.context_utilization  = _clamp(result.get("context_utilization", 5))
    rs.gemini.response_quality     = _clamp(result.get("response_quality", 5))
    rs.gemini.factual_accuracy     = _clamp(result.get("factual_accuracy", 7))
    rs.gemini.context_leverage     = _clamp(result.get("context_leverage", 5))
    rs.gemini.reasoning            = str(result.get("reasoning", ""))
    rs.gemini.available            = True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def parse_context_score(context_str: str) -> Tuple[float, str]:
    match = re.search(r"score:\s*([-\d.]+)", str(context_str))
    score = float(match.group(1)) if match else 0.0
    text_match = re.search(r"text:\s*'(.+)'", str(context_str))
    text = text_match.group(1) if text_match else ""
    return score, text


def evaluate(
    csv_path: str,
    gemini_api_key: Optional[str] = None,
    heuristic_weight: float = 0.4,
    gemini_weight: float = 0.6,
) -> List[ResponseScore]:
    df = pd.read_csv(csv_path)
    use_gemini = bool(gemini_api_key)
    total = len(df)

    if use_gemini:
        print(f"\n  Gemini LLM-as-Judge ENABLED  (model: {GEMINI_MODEL})")
        print(f"  Blend: {int(heuristic_weight*100)}% heuristic + {int(gemini_weight*100)}% Gemini")
        print(f"  Scoring {total} responses -- this will take ~{total * (RATE_LIMIT_WAIT+2)}s...\n")
    else:
        print(f"\n  Gemini DISABLED -- running heuristic-only scoring.")
        print("  (Set GEMINI_API_KEY env var or pass --gemini-key to enable Gemini.)\n")

    records = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        question = row["Question (1-5)"]
        ctx_score, ctx_text = parse_context_score(row["Context pulled, score and text"])
        rs = ResponseScore(
            model=row["Model name"],
            tuned=row["Tuned/untuned or base"],
            question=question,
            qtype=QUESTION_TYPES.get(question, "unknown"),
            context_score=ctx_score,
            context_text=ctx_text,
            answer=str(row["Answer"]),
            heuristic_weight=heuristic_weight,
            gemini_weight=gemini_weight,
        )
        run_heuristics(rs)
        if use_gemini:
            print(f"  [{i:02d}/{total}] {rs.model} [{rs.tuned}] -- {question[:45]}...")
            score_with_gemini(rs, gemini_api_key)
            if rs.gemini.available:
                print(f"         CLS={rs.gemini.context_leverage:.1f}  FA={rs.gemini.factual_accuracy:.1f}  {rs.gemini.reasoning[:80]}")
            time.sleep(RATE_LIMIT_WAIT)
        records.append(rs)
    return records


def records_to_df(records: List[ResponseScore]) -> pd.DataFrame:
    rows = []
    for r in records:
        b = r.blended
        row = {
            "Model":               r.model,
            "Variant":             r.tuned,
            "Model+Variant":       f"{r.model} [{r.tuned}]",
            "Question":            r.question,
            "Q_Type":              r.qtype,
            "Context_Score":       round(r.context_score, 3),
            "Answer_WordCount":    len(r.answer.split()),
            # Heuristic
            "H_RA":  round(r.heuristic.role_adherence, 1),
            "H_BE":  round(r.heuristic.boundary_enforcement, 1),
            "H_AR":  round(r.heuristic.answer_relevance, 1),
            "H_CU":  round(r.heuristic.context_utilization, 1),
            "H_RQ":  round(r.heuristic.response_quality, 1),
            "H_FA":  round(r.heuristic.factual_accuracy, 1),
            "H_CLS": round(r.heuristic.context_leverage, 1),
            "Heuristic_Composite": round(r.heuristic_composite, 2),
            # Gemini
            "G_RA":  round(r.gemini.role_adherence, 1)       if r.gemini.available else None,
            "G_BE":  round(r.gemini.boundary_enforcement, 1) if r.gemini.available else None,
            "G_AR":  round(r.gemini.answer_relevance, 1)     if r.gemini.available else None,
            "G_CU":  round(r.gemini.context_utilization, 1)  if r.gemini.available else None,
            "G_RQ":  round(r.gemini.response_quality, 1)     if r.gemini.available else None,
            "G_FA":  round(r.gemini.factual_accuracy, 1)     if r.gemini.available else None,
            "G_CLS": round(r.gemini.context_leverage, 1)     if r.gemini.available else None,
            "Gemini_Composite": round(r.gemini_composite, 2) if r.gemini.available else None,
            "Gemini_Reasoning": r.gemini.reasoning            if r.gemini.available else "",
            # Blended final
            "RA":  round(b["role_adherence"], 1),
            "BE":  round(b["boundary_enforcement"], 1),
            "AR":  round(b["answer_relevance"], 1),
            "CU":  round(b["context_utilization"], 1),
            "RQ":  round(b["response_quality"], 1),
            "FA":  round(b["factual_accuracy"], 1),
            "CLS": round(b["context_leverage"], 1),
            "Composite_Score": round(r.composite_score, 2),
            "Gemini_Used": r.gemini.available,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_comparative_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Overall rank per question
    df["Q_Rank"] = (
        df.groupby("Question")["Composite_Score"]
        .rank(ascending=False, method="min").astype(int)
    )
    df["Best_Answer"] = (
        df.groupby("Question")["Composite_Score"].transform("max") == df["Composite_Score"]
    )
    # Comparative Context Leverage rank per question
    df["CCL_Rank"] = (
        df.groupby("Question")["CLS"]
        .rank(ascending=False, method="min").astype(int)
    )
    df["CCL_Delta_From_Best"] = (
        df.groupby("Question")["CLS"].transform("max") - df["CLS"]
    ).round(2)
    # H vs G agreement
    mask = df["Gemini_Composite"].notna()
    df["H_G_Agreement"] = None
    df.loc[mask, "H_G_Agreement"] = (
        (df.loc[mask, "Heuristic_Composite"] - df.loc[mask, "Gemini_Composite"]).abs().round(1)
    )
    return df


def tuning_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Composite tuning delta and CLS-specific tuning delta."""
    base = df.reset_index(drop=True)

    comp_pivot = base.pivot_table(
        index=["Model", "Question"], columns="Variant",
        values="Composite_Score", aggfunc="first",
    ).reset_index()
    comp_pivot.columns.name = None
    if "Tuned" in comp_pivot.columns and "Base" in comp_pivot.columns:
        comp_pivot["Tuning_Delta"] = comp_pivot["Tuned"] - comp_pivot["Base"]
    else:
        comp_pivot["Tuning_Delta"] = float("nan")

    cls_pivot = base.pivot_table(
        index=["Model", "Question"], columns="Variant",
        values="CLS", aggfunc="first",
    ).reset_index()
    cls_pivot.columns.name = None
    rename = {}
    if "Tuned" in cls_pivot.columns:
        rename["Tuned"] = "CLS_Tuned"
    if "Base" in cls_pivot.columns:
        rename["Base"] = "CLS_Base"
    cls_pivot = cls_pivot.rename(columns=rename)
    if "CLS_Tuned" in cls_pivot.columns and "CLS_Base" in cls_pivot.columns:
        cls_pivot["CLS_Tuning_Delta"] = cls_pivot["CLS_Tuned"] - cls_pivot["CLS_Base"]
    else:
        cls_pivot["CLS_Tuning_Delta"] = float("nan")

    merged = comp_pivot.merge(
        cls_pivot[["Model", "Question"] + [c for c in cls_pivot.columns if c.startswith("CLS")]],
        on=["Model", "Question"], how="left",
    )
    return merged


def overall_ranking(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = dict(
        Mean_Composite    =("Composite_Score", "mean"),
        Min_Score         =("Composite_Score", "min"),
        Max_Score         =("Composite_Score", "max"),
        Std_Dev           =("Composite_Score", "std"),
        Mean_RA           =("RA",  "mean"),
        Mean_BE           =("BE",  "mean"),
        Mean_AR           =("AR",  "mean"),
        Mean_CU           =("CU",  "mean"),
        Mean_RQ           =("RQ",  "mean"),
        Mean_FA           =("FA",  "mean"),
        Mean_CLS          =("CLS", "mean"),
        Best_Answer_Count =("Best_Answer", "sum"),
        H_Mean            =("Heuristic_Composite", "mean"),
    )
    if df["Gemini_Composite"].notna().any():
        agg_cols["G_Mean"] = ("Gemini_Composite", "mean")
    agg = (
        df.groupby("Model+Variant").agg(**agg_cols)
        .reset_index().sort_values("Mean_Composite", ascending=False)
    )
    agg["Overall_Rank"] = range(1, len(agg) + 1)
    return agg.round(2)


# ---------------------------------------------------------------------------
# Report builder — writes to both stdout and a buffer for .txt
# ---------------------------------------------------------------------------

class Tee:
    """Write to both a stream and an internal buffer simultaneously."""
    def __init__(self, stream):
        self.stream = stream
        self.buf = io.StringIO()

    def write(self, msg):
        self.stream.write(msg)
        self.buf.write(msg)

    def flush(self):
        self.stream.flush()

    def getvalue(self):
        return self.buf.getvalue()


def build_report(df, ranking, impact, gemini_enabled, h_weight, g_weight) -> str:
    tee = Tee(sys.stdout)
    orig_stdout = sys.stdout
    sys.stdout = tee

    sep  = "=" * 88
    sep2 = "-" * 88

    print("\n" + sep)
    print("  W&M CHATBOT MODEL EVALUATION REPORT  (v3)")
    print(sep)
    mode = (f"Gemini LLM-as-Judge + Heuristic ({int(h_weight*100)}% / {int(g_weight*100)}%)"
            if gemini_enabled else "Heuristic-only")
    print(f"  Scoring mode : {mode}")
    print(f"  Models       : Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, gemma-2-9b-it")
    print(f"  Variants     : Tuned, Base  (6 model+variant combinations)")
    print(f"  Questions    : 5  (factual_wm x2, guidance_wm x1, off_topic x1, policy_violation x1)")
    print()
    print("  DIMENSION WEIGHTS:")
    for dim, w in DIM_WEIGHTS.items():
        notes = {
            "factual_accuracy": " <- Gemini judges hallucinations best",
            "context_leverage": " <- NEW: question-type-aware context extraction + verbosity penalty",
            "response_quality": " <- Verbosity penalty scaled to question type",
        }
        print(f"    {dim:<24} {int(w*100):>3}%{notes.get(dim, '')}")

    # ── Per-question scores ────────────────────────────────────────────────
    print()
    print(sep)
    print("  PER-QUESTION SCORES")
    print(sep)
    hdr = f"   {'Model+Variant':<40}  RA   BE   AR   CU   RQ   FA  CLS  Comp  Rank  CCL"
    for q in df["Question"].unique():
        sub = df[df["Question"] == q].sort_values("Composite_Score", ascending=False)
        qtype = sub.iloc[0]["Q_Type"]
        ctx   = sub.iloc[0]["Context_Score"]
        lo, hi = VERBOSITY_TARGETS.get(qtype, (40, 200))
        print(f"\nQ: {q}")
        print(f"   Type: {qtype}  |  Context similarity: {ctx:.3f}  |  Ideal length: {lo}-{hi} words")
        print(hdr)
        print(f"   {sep2}")
        for _, row in sub.iterrows():
            best = " *" if row.get("Best_Answer", False) else "  "
            ccl_gap = row.get("CCL_Delta_From_Best", 0)
            ccl_str = f"+{ccl_gap:.1f}" if ccl_gap > 0 else "best"
            print(
                f"   {row['Model+Variant']:<40} "
                f"{row['RA']:>4.1f} {row['BE']:>4.1f} {row['AR']:>4.1f} "
                f"{row['CU']:>4.1f} {row['RQ']:>4.1f} {row['FA']:>4.1f} "
                f"{row['CLS']:>4.1f}  {row['Composite_Score']:>5.1f}"
                f"   #{row['Q_Rank']}{best}  {ccl_str}"
            )
            if gemini_enabled and row.get("Gemini_Reasoning"):
                wrapped = textwrap.fill(
                    row["Gemini_Reasoning"], width=76,
                    initial_indent="         Gemini: ",
                    subsequent_indent="                 ",
                )
                print(wrapped)

    # ── CCL summary ────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  COMPARATIVE CONTEXT LEVERAGE (CCL)  -- who extracted context best per question")
    print(sep)
    print(f"   {'Question':<48} {'Best Model+Variant':<42} CLS   Gap")
    print(f"   {sep2}")
    for q in df["Question"].unique():
        sub = df[df["Question"] == q]
        best_row = sub.loc[sub["CLS"].idxmax()]
        worst_row = sub.loc[sub["CLS"].idxmin()]
        gap = best_row["CLS"] - worst_row["CLS"]
        print(f"   {q[:47]:<48} {best_row['Model+Variant']:<42} {best_row['CLS']:>4.1f}  gap={gap:.1f}")

    # ── H vs G agreement ─────────────────────────────────────────────────
    if gemini_enabled and df["H_G_Agreement"].notna().any():
        print()
        print(sep)
        print("  HEURISTIC vs GEMINI AGREEMENT  (mean |H-G| composite per model+variant)")
        print(sep)
        agree = (
            df[df["H_G_Agreement"].notna()]
            .groupby("Model+Variant")["H_G_Agreement"].mean().sort_values()
        )
        print(f"   {'Model+Variant':<44} {'|H-G|':>6}   Interpretation")
        print(f"   {sep2}")
        for mv, gap in agree.items():
            interp = ("High agreement" if gap < 5 else
                      "Moderate agreement" if gap < 10 else "Significant divergence")
            print(f"   {mv:<44} {gap:>6.1f}   {interp}")

    # ── Tuning impact ──────────────────────────────────────────────────────
    print()
    print(sep)
    print("  FINE-TUNING IMPACT  (Tuned - Base blended composite per question)")
    print(sep)
    print(f"   {'Model':<34} {'Question':<46} Composite  CLS_Delta")
    print(f"   {sep2}")
    for _, row in impact.iterrows():
        d  = row.get("Tuning_Delta", float("nan"))
        cd = row.get("CLS_Tuning_Delta", float("nan"))
        print(f"   {row['Model']:<34} {str(row['Question'])[:45]:<46} {d:>+7.2f}    {cd:>+6.2f}")

    # ── Overall ranking ────────────────────────────────────────────────────
    print()
    print(sep)
    print("  OVERALL RANKING  (mean blended composite across 5 questions)")
    print(sep)
    g_col = "  G_Score" if "G_Mean" in ranking.columns else ""
    print(f"   {'Rk':<4} {'Model+Variant':<44} {'Score':>6} {'H':>6}{g_col}  {'*':>3}  Dimension means")
    print(f"   {sep2}")
    for _, row in ranking.iterrows():
        g_str = f"  {row['G_Mean']:>5.1f}" if "G_Mean" in ranking.columns else ""
        print(
            f"   #{row['Overall_Rank']:<3}"
            f" {row['Model+Variant']:<44}"
            f" {row['Mean_Composite']:>6.2f}"
            f"  {row['H_Mean']:>5.1f}"
            f"{g_str}"
            f"  {int(row['Best_Answer_Count']):>3}"
            f"   RA:{row['Mean_RA']:.1f} BE:{row['Mean_BE']:.1f} "
            f"AR:{row['Mean_AR']:.1f} CU:{row['Mean_CU']:.1f} "
            f"RQ:{row['Mean_RQ']:.1f} FA:{row['Mean_FA']:.1f} CLS:{row['Mean_CLS']:.1f}"
        )

    # ── Tuning summary ─────────────────────────────────────────────────────
    print()
    print(sep)
    print("  FINE-TUNING SUMMARY")
    print(sep)
    for model, delta in impact.groupby("Model")["Tuning_Delta"].mean().sort_values(ascending=False).items():
        cls_delta = impact.groupby("Model")["CLS_Tuning_Delta"].mean().get(model, float("nan"))
        verdict = ("fine-tuning helped" if delta > 0.5 else
                   "fine-tuning hurt"   if delta < -0.5 else "negligible effect")
        cls_note = (f"  CLS delta: {cls_delta:+.2f}" if not np.isnan(cls_delta) else "")
        print(f"   {model:<34}  composite {delta:+.2f}  -> {verdict}{cls_note}")

    # ── Narrative interpretation ────────────────────────────────────────────
    print()
    print(sep)
    print("  NARRATIVE INTERPRETATION")
    print(sep)

    top_mv    = ranking.iloc[0]["Model+Variant"]
    bot_mv    = ranking.iloc[-1]["Model+Variant"]
    top_score = ranking.iloc[0]["Mean_Composite"]
    bot_score = ranking.iloc[-1]["Mean_Composite"]

    best_cls_mv  = ranking.loc[ranking["Mean_CLS"].idxmax(),  "Model+Variant"]
    worst_cls_mv = ranking.loc[ranking["Mean_CLS"].idxmin(),  "Model+Variant"]
    best_fa_mv   = ranking.loc[ranking["Mean_FA"].idxmax(),   "Model+Variant"]
    best_be_mv   = ranking.loc[ranking["Mean_BE"].idxmax(),   "Model+Variant"]

    avg_deltas = impact.groupby("Model")[["Tuning_Delta", "CLS_Tuning_Delta"]].mean()

    print(f"""
  OVERALL PERFORMANCE
  {top_mv} ranked first with a mean composite of {top_score:.1f}/100.
  {bot_mv} ranked last at {bot_score:.1f}/100 — a gap of {top_score-bot_score:.1f} points.

  CONTEXT LEVERAGE (CLS)
  {best_cls_mv} extracted the most value from retrieved context, suggesting it is
  most effective at grounding responses in W&M-specific information.
  {worst_cls_mv} struggled most with context leverage — either ignoring relevant
  context or producing verbose responses where conciseness was required.

  FACTUAL ACCURACY
  {best_fa_mv} had the highest factual accuracy score, indicating it is least
  likely to hallucinate W&M-specific details like dates, program names, or policies.

  BOUNDARY ENFORCEMENT
  {best_be_mv} handled off-topic and policy-violating questions most effectively,
  declining clearly while redirecting students to appropriate W&M resources.

  FINE-TUNING EFFECTS""")

    for model in avg_deltas.index:
        d   = avg_deltas.loc[model, "Tuning_Delta"]
        cld = avg_deltas.loc[model, "CLS_Tuning_Delta"]
        if d > 0.5:
            verdict = f"Fine-tuning improved {model} overall (+{d:.2f}) and on context leverage ({cld:+.2f})."
        elif d < -0.5:
            verdict = (f"Fine-tuning hurt {model} overall ({d:.2f}). "
                       f"Context leverage changed by {cld:+.2f} — "
                       + ("also degraded." if cld < -0.3 else "partially maintained."))
        else:
            verdict = f"Fine-tuning had negligible effect on {model} ({d:+.2f} composite, {cld:+.2f} CLS)."
        print(f"  - {verdict}")

    print()

    sys.stdout = orig_stdout
    return tee.getvalue()


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

DIMS = ["RA", "BE", "AR", "CU", "RQ", "FA", "CLS"]
DIM_LABELS = {
    "RA": "Role\nAdherence", "BE": "Boundary\nEnforce",
    "AR": "Answer\nRelevance", "CU": "Context\nUtil",
    "RQ": "Response\nQuality", "FA": "Factual\nAccuracy",
    "CLS": "Context\nLeverage",
}

def _model_color(mv: str) -> str:
    for model, color in MODEL_COLORS.items():
        if model in mv:
            return color
    return "#888888"

def _hatch(mv: str) -> str:
    return "//" if "[Base]" in mv else ""


def plot_overall_composite(ranking: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    mvs    = ranking["Model+Variant"].tolist()[::-1]
    scores = ranking["Mean_Composite"].tolist()[::-1]
    colors = [_model_color(mv) for mv in mvs]
    hatches = [_hatch(mv) for mv in mvs]

    bars = ax.barh(mvs, scores, color=colors, edgecolor="white", height=0.6)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{score:.1f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Mean Composite Score (0–100)", fontsize=11)
    ax.set_title("Overall Composite Scores — All Model Variants", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlim(0, 105)
    ax.axvline(x=scores[0], color="gray", linestyle="--", alpha=0.3)

    # Legend
    patches = [mpatches.Patch(facecolor=c, label=m.replace("-Instruct", "").replace("-v0.3",""))
               for m, c in MODEL_COLORS.items()]
    tuned_patch = mpatches.Patch(facecolor="white", edgecolor="black", label="Tuned (solid)")
    base_patch  = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Base (hatched)")
    ax.legend(handles=patches + [tuned_patch, base_patch], loc="lower right", fontsize=9)

    sns.despine(left=True)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def plot_radar(ranking: pd.DataFrame, out_path: str) -> None:
    mvs = ranking["Model+Variant"].tolist()
    n_dims = len(DIMS)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    cols = [f"Mean_{d}" if f"Mean_{d}" in ranking.columns else d for d in DIMS]
    # Map dimension names to ranking column names
    dim_col_map = {
        "RA": "Mean_RA", "BE": "Mean_BE", "AR": "Mean_AR",
        "CU": "Mean_CU", "RQ": "Mean_RQ", "FA": "Mean_FA", "CLS": "Mean_CLS",
    }

    n_models = len(mvs)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows),
                             subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for i, (mv, ax) in enumerate(zip(mvs, axes)):
        row = ranking[ranking["Model+Variant"] == mv].iloc[0]
        values = [row[dim_col_map[d]] for d in DIMS]
        values += values[:1]

        color = _model_color(mv)
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([DIM_LABELS[d] for d in DIMS], size=8)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(["2", "4", "6", "8", "10"], size=6, color="gray")
        ax.set_title(mv.replace("-Instruct", "").replace("-v0.3", ""),
                     size=9, fontweight="bold", pad=12)
        ax.grid(color="gray", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Dimension Profiles — All Model Variants", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def plot_tuning_delta(impact: pd.DataFrame, out_path: str) -> None:
    models    = impact["Model"].unique()
    questions = [q[:35] + "…" if len(q) > 35 else q for q in impact["Question"].unique()]
    q_full    = impact["Question"].unique()

    x      = np.arange(len(questions))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models):
        sub    = impact[impact["Model"] == model].set_index("Question")
        deltas = [sub.loc[q, "Tuning_Delta"] if q in sub.index else 0 for q in q_full]
        color  = MODEL_COLORS.get(model, "#888888")
        bars   = ax.bar(x + i * width, deltas, width, label=model.replace("-Instruct","").replace("-v0.3",""),
                        color=color, edgecolor="white")
        for bar, d in zip(bars, deltas):
            if not np.isnan(d):
                ax.text(bar.get_x() + bar.get_width()/2, d + (0.2 if d >= 0 else -0.8),
                        f"{d:+.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(questions, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Score Delta (Tuned − Base)", fontsize=11)
    ax.set_title("Fine-Tuning Impact per Question  (positive = tuning helped)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def plot_cls_heatmap(df: pd.DataFrame, out_path: str) -> None:
    pivot = df.pivot_table(index="Model+Variant", columns="Question", values="CLS", aggfunc="mean")
    # Shorten question labels
    pivot.columns = [c[:40] + "…" if len(c) > 40 else c for c in pivot.columns]
    # Shorten model labels
    pivot.index = [i.replace("-Instruct","").replace("-v0.3","") for i in pivot.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=0, vmax=10, linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "CLS Score (0–10)"})
    ax.set_title("Comparative Context Leverage (CLS) — Model vs Question",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def plot_cls_tuning_delta(impact: pd.DataFrame, out_path: str) -> None:
    if "CLS_Tuning_Delta" not in impact.columns:
        return
    models    = impact["Model"].unique()
    q_full    = impact["Question"].unique()
    questions = [q[:35] + "…" if len(q) > 35 else q for q in q_full]

    x      = np.arange(len(questions))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models):
        sub    = impact[impact["Model"] == model].set_index("Question")
        deltas = [sub.loc[q, "CLS_Tuning_Delta"] if q in sub.index else 0 for q in q_full]
        color  = MODEL_COLORS.get(model, "#888888")
        bars   = ax.bar(x + i * width, deltas, width,
                        label=model.replace("-Instruct","").replace("-v0.3",""),
                        color=color, edgecolor="white", alpha=0.85)
        for bar, d in zip(bars, deltas):
            if not np.isnan(d):
                ax.text(bar.get_x() + bar.get_width()/2, d + (0.1 if d >= 0 else -0.5),
                        f"{d:+.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(questions, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("CLS Delta (Tuned − Base)", fontsize=11)
    ax.set_title("Fine-Tuning Effect on Context Leverage (CLS)\n(positive = tuning improved context extraction)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def generate_all_graphs(df, ranking, impact, results_dir) -> None:
    print("\n  Generating graphs...")
    plot_overall_composite(ranking, os.path.join(results_dir, "01_overall_composite_scores.png"))
    plot_radar(ranking,             os.path.join(results_dir, "02_dimension_radar.png"))
    plot_tuning_delta(impact,       os.path.join(results_dir, "03_tuning_delta.png"))
    plot_cls_heatmap(df,            os.path.join(results_dir, "04_cls_heatmap.png"))
    plot_cls_tuning_delta(impact,   os.path.join(results_dir, "05_tuned_vs_base_cls_delta.png"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="W&M chatbot model evaluator — Gemini LLM-as-Judge + CLS + graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python evaluate_models.py data.csv
              GEMINI_API_KEY=AIza... python evaluate_models.py data.csv
              python evaluate_models.py data.csv --gemini-key AIza... --heuristic-weight 0.3 --gemini-weight 0.7
              python evaluate_models.py data.csv --no-gemini
        """),
    )
    p.add_argument("csv")
    p.add_argument("--gemini-key", default=None)
    p.add_argument("--no-gemini", action="store_true")
    p.add_argument("--heuristic-weight", type=float, default=0.4)
    p.add_argument("--gemini-weight",    type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()
    if abs(args.heuristic_weight + args.gemini_weight - 1.0) > 1e-6:
        print("ERROR: weights must sum to 1.0", file=sys.stderr)
        sys.exit(1)

    api_key = None
    if not args.no_gemini:
        api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")

    # Results directory next to the script
    here        = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Evaluate
    records = evaluate(args.csv, gemini_api_key=api_key,
                       heuristic_weight=args.heuristic_weight,
                       gemini_weight=args.gemini_weight)

    gemini_enabled = any(r.gemini.available for r in records)
    df      = records_to_df(records)
    df      = add_comparative_columns(df)
    impact  = tuning_impact(df)
    ranking = overall_ranking(df)

    # Print report + capture to string
    report_text = build_report(df, ranking, impact, gemini_enabled,
                               args.heuristic_weight, args.gemini_weight)

    # Save outputs
    csv_path    = os.path.join(results_dir, "model_evaluation_scored.csv")
    report_path = os.path.join(results_dir, "summary_report.txt")

    df.to_csv(csv_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    generate_all_graphs(df, ranking, impact, results_dir)

    print(f"\n  All outputs saved to: {results_dir}/")
    print(f"    summary_report.txt")
    print(f"    model_evaluation_scored.csv")
    print(f"    01_overall_composite_scores.png")
    print(f"    02_dimension_radar.png")
    print(f"    03_tuning_delta.png")
    print(f"    04_cls_heatmap.png")
    print(f"    05_tuned_vs_base_cls_delta.png\n")


if __name__ == "__main__":
    main()