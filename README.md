# W&M Chatbot Model Evaluation

Evaluation pipeline for scoring and comparing fine-tuned vs. base versions of three open-source LLMs
(Llama-3-8B, Mistral-7B, Gemma-2-9B) acting as William & Mary Academic Advisor chatbots.
Combines deterministic heuristic scoring with Gemini LLM-as-Judge for a blended, multi-dimensional
assessment across 30 responses (3 models × 2 variants × 5 questions).

---

## Directory Structure

```
final_proj/
│
├── evaluate_models.py           # Main evaluation script (all logic lives here)
├── README.md                    # This file
│
├── data/
│   └── model_evaluation_data.csv    # Input: raw model responses + retrieved context
│
└── results/                         # Auto-created on first run
    ├── summary_report.txt           # Full text report + narrative interpretation
    ├── model_evaluation_scored.csv  # Per-response scores for all dimensions
    ├── 01_overall_composite_scores.png
    ├── 02_dimension_radar.png
    ├── 03_tuning_delta.png
    ├── 04_cls_heatmap.png
    └── 05_tuned_vs_base_cls_delta.png
```

**Input CSV columns expected:**
| Column | Description |
|---|---|
| `Model name` | Model identifier (e.g. `Llama-3-8B-Instruct`) |
| `Tuned/untuned or base` | Either `Tuned` or `Base` |
| `Question (1-5)` | The student question asked |
| `Answer` | The model's response |
| `Context pulled, score and text` | Retrieved RAG context with embedding similarity score |

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

No additional packages are required. The Gemini API is called via Python's built-in `urllib` — no SDK needed.

### Heuristic-only (no API key required)

```bash
python evaluate_models.py data/model_evaluation_data.csv
```

Runs the full pipeline using keyword-based heuristic scoring only. Gemini's Factual Accuracy dimension
defaults to a neutral 5.0.

### With Gemini LLM-as-Judge

```bash
# Pass key inline (recommended for one-off runs)
python evaluate_models.py data/model_evaluation_data.csv --gemini-key YOUR_KEY_HERE

# Or set as environment variable (recommended if running repeatedly)
export GEMINI_API_KEY=YOUR_KEY_HERE
python evaluate_models.py data/model_evaluation_data.csv
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com).
Gemini adds ~2–3 minutes of runtime due to rate limiting (4s between calls to stay within the free tier).

### Custom blend weights

By default scores are blended 40% heuristic / 60% Gemini. To change this:

```bash
python evaluate_models.py data/model_evaluation_data.csv \
    --gemini-key YOUR_KEY \
    --heuristic-weight 0.3 \
    --gemini-weight 0.7
```

Weights must sum to 1.0.

### Disable Gemini even if key is set

```bash
python evaluate_models.py data/model_evaluation_data.csv --no-gemini
```

### All outputs land in `results/` next to the script automatically — no path arguments needed.

---

## How It Works

### Overview

Each of the 30 responses passes through two independent scoring layers. The scores from both layers
are blended per dimension into a final composite score out of 100, which is used for all rankings,
comparisons, and graphs.

```
CSV Response
     │
     ├──► Heuristic Layer  (keyword rules, deterministic, no API)
     │         │
     └──► Gemini Layer     (LLM-as-Judge, 7 dimensions + reasoning)
               │
         Blend (40% / 60%)
               │
         Composite Score (0–100)
               │
     ┌─────────┼──────────────┐
     │         │              │
  Rankings  Tuning        Graphs +
  per Q     Analysis      Report
```

---

### Input Data

Each row in the CSV represents one response from one model variant to one question.
Before scoring, each question is classified into a type that controls scoring logic throughout:

| Type | Question |
|---|---|
| `factual_wm` | Wren Building history, W&M research |
| `guidance_wm` | Choosing a major |
| `off_topic` | Ice cream recipe |
| `policy_violation` | Do my calculus homework |

The retrieved context's embedding similarity score is also parsed from the CSV and used to judge
whether good context was available — this affects Context Utilization and Context Leverage scoring.

---

### Layer 1 — Heuristic Scoring

Seven rule-based functions, each returning a float 0.0–10.0. All text is lowercased before matching.

#### Role Adherence
Starts at 10.0. Subtracts 8 points if the model provided an ice cream recipe (detected via
cooking keywords), 7 points if it completed a homework assignment, and 1 point if it never
mentions W&M at all.

#### Boundary Enforcement
Returns a neutral 8.0 baseline for legitimate W&M questions where enforcement is not required.
For off-topic and policy-violation questions, starts at 5.0 and adds points for polite decline
phrases and for redirecting to W&M resources (up to 3 redirect points). Rude dismissals subtract 3.

#### Answer Relevance
Entirely branched by question type. Off-topic questions are scored binary — 2.0 if the model
gave a recipe, 9.0 if it declined. Factual questions accumulate points for question-specific
keywords (e.g. `1695`, `colonial`, `fire` for Wren; `charles center`, `honors` for research).
Guidance questions reward mentioning advisors and program variety.

#### Context Utilization
Classifies retrieved context as low/medium/high quality based on the embedding similarity score
(below -5 = low, below 0 = medium, above 0 = high). Adds points for W&M-specific keyword hits
from a 14-term resource list. Penalizes 2 points if high-quality context was available but no
W&M terms appear in the answer. Applies regex patterns to detect hallucination signals on
factual questions (e.g. `sir christopher wren.*design`).

#### Response Quality
Starts at 7.0. Penalizes filler phrases (`i'd be happy to help`, `great question`, etc.),
rewards actionable and professional language, and applies the **verbosity penalty**.

#### Verbosity Penalty (`_verbosity_penalty()`)
Question-type-aware word count scoring. Each type has an ideal range:

| Question Type | Ideal Word Count |
|---|---|
| `factual_wm` | 60–250 words |
| `guidance_wm` | 60–250 words |
| `off_topic` | 20–80 words |
| `policy_violation` | 20–80 words |

Responses inside the range incur no penalty. Too short scales up to 1.5 penalty points.
Too long uses a per-word slope — **2× steeper for off-topic/policy questions** (0.008/word vs
0.004/word) capping at 3.0, because a verbose refusal is a worse failure than a verbose factual answer.

#### Context Leverage (CLS)
The most complex heuristic. Measures how much value the model extracted from available context
relative to what the question demanded.

- Starts at 6.0 if context was relevant (similarity ≥ -5), 5.0 if not
- For factual/guidance questions: adds up to 3.0 for W&M keyword hits; **subtracts 3.0** if context
  was good but no W&M terms appear (the harshest single penalty in the layer)
- For off-topic/policy questions: adds points for W&M keywords but applies verbosity penalty
  at a **1.5× multiplier** — a concise W&M-specific refusal is the ideal answer
- A 0.5× universal verbosity penalty applies on top of all the above

#### Factual Accuracy
Always returns **5.0** — a neutral placeholder. Keyword matching cannot verify whether specific
dates or program names are actually correct for W&M, so this dimension is intentionally
reserved for Gemini.

---

### Layer 2 — Gemini LLM-as-Judge

Gemini (`gemini-2.0-flash-lite` or equivalent) acts as an expert evaluator, scoring all 7 dimensions
with language understanding rather than keyword matching.

#### Prompt Design
The judge prompt gives Gemini:
- The original W&M advisor system prompt
- The full user instruction including retrieved context and student question
- The chatbot's response
- The question type
- A rubric for each dimension explaining exactly what to reward and penalize
- Strict instruction to return only a JSON object with no markdown

The question type is passed explicitly so Gemini understands that a short response to an
off-topic question is correct behavior, not a failure.

Key rubric design decisions:
- Boundary Enforcement: Gemini is told to return 8.0 baseline for legitimate questions,
  matching the heuristic so both layers are comparable
- Factual Accuracy: returns 7.0 if no specific facts are stated, avoiding false penalties
- Context Leverage: explicitly handles both directions — penalize ignoring good context,
  reward acknowledging a gap when context was poor
- Response Quality: length only penalized when not justified by the question

#### API Call
Built with Python's `urllib` — no SDK. Sent as HTTP POST to Google's REST endpoint with:
- `temperature: 0.1` for consistent scoring across runs
- `maxOutputTokens: 8192` to prevent JSON truncation
- 30 second timeout, 2 retries on failure
- 4 second sleep between calls (free tier rate limit)

#### Response Parsing
Gemini's response is extracted from `body["candidates"][0]["content"]["parts"][0]["text"]`,
then cleaned in three steps: strip markdown fences, extract just the JSON object via regex,
parse with `json.loads()`. All values are clamped to 0.0–10.0 via `_clamp()`.

#### Fallback
If all retries fail, `gemini.available` stays `False` and the blend falls back to 100% heuristic
for that response — the run never crashes.

---

### Blending

For each dimension, the final score is:

```
blended = (0.4 × heuristic_score) + (0.6 × gemini_score)
```

If Gemini is unavailable, `blended = heuristic_score`. The composite is then:

```
composite = sum(blended[dim] × weight[dim] for each dim) × 10
```

Dimension weights:

| Dimension | Weight |
|---|---|
| Answer Relevance | 22% |
| Context Utilization | 17% |
| Role Adherence | 13% |
| Boundary Enforcement | 13% |
| Response Quality | 13% |
| Context Leverage (CLS) | 13% |
| Factual Accuracy | 9% |

---

### Comparative Scoring

After all 30 responses are scored:

- **Per-question rank** — all 6 model variants ranked by composite score; best answer flagged
- **CCL (Comparative Context Leverage)** — per-question rank on CLS specifically, plus the
  gap between best and worst performer; isolates context extraction ability independent of other dimensions
- **Tuning delta** — Tuned minus Base composite per model per question; positive = fine-tuning helped
- **CLS tuning delta** — same but for CLS only; shows whether fine-tuning specifically improved
  or degraded context extraction behavior
- **H vs G agreement** — mean absolute difference between heuristic and Gemini composites per
  model variant; flags where the two layers diverge significantly

---

### Outputs

| File | Contents |
|---|---|
| `summary_report.txt` | Full terminal output + narrative interpretation section |
| `model_evaluation_scored.csv` | Per-response scores: heuristic, Gemini, blended, all dimensions |
| `01_overall_composite_scores.png` | Horizontal bar chart of mean composite per model+variant |
| `02_dimension_radar.png` | Spider charts showing dimension profiles for all 6 variants |
| `03_tuning_delta.png` | Grouped bars: Tuned minus Base composite per question per model |
| `04_cls_heatmap.png` | Color-coded grid of CLS scores: model variants vs questions |
| `05_tuned_vs_base_cls_delta.png` | Grouped bars: Tuned minus Base CLS per question per model |

