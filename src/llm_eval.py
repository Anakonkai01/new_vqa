"""LLM-as-a-Judge evaluation using Gemini.

Scores VQA model predictions on a 0–5 scale by prompting a Gemini model to
compare each prediction against the ground-truth answer + explanation.

Rubric
------
* 0 — Completely wrong answer and reasoning.
* 1 — Captures a small fragment of the truth but is mostly incorrect.
* 2 — Partially correct but misses key reasoning or details.
* 3 — Mostly correct meaning; reasoning slightly flawed or missing a detail.
* 4 — Correct meaning and reasoning, phrased differently or less comprehensively.
* 5 — Perfect match in meaning and reasoning.

Input format
------------
The ``--input_json`` file must be a JSON array where each element is a dict
with the keys ``question``, ``ground_truth``, and ``prediction``::

    [
        {
            "question": "What color is the umbrella?",
            "ground_truth": "The umbrella is red because ...",
            "prediction": "red because the sky is overcast ..."
        },
        ...
    ]

Rate limiting
-------------
A 4-second sleep is inserted after every successful API call to respect the
free-tier quota (≈15 RPM).  A single retry with a 10-second backoff is
attempted on failure before falling back to score 0.

Usage
-----
    export GEMINI_API_KEY='your_api_key_here'
    python src/llm_eval.py \\
        --input_json results/model_e_predictions.json \\
        --output_json results/llm_scores.json \\
        --model gemini-1.5-flash \\
        --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import google.generativeai as genai
from tqdm import tqdm


# ── Setup ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate VQA model predictions with LLM-as-a-Judge (Gemini)."
    )
    parser.add_argument(
        "--input_json", type=str, required=True,
        help="Path to JSON file with model predictions. Each element must have "
             "'question', 'ground_truth', and 'prediction' keys.",
    )
    parser.add_argument(
        "--output_json", type=str, default="llm_evaluation_results.json",
        help="Path to save evaluation results with per-sample LLM scores.",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-1.5-flash",
        help="Gemini model version to use (default: gemini-1.5-flash).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of samples to evaluate (useful for testing).",
    )
    return parser.parse_args()


def _setup_gemini() -> None:
    """Configure the Gemini API client from the ``GEMINI_API_KEY`` env variable.

    Raises:
        SystemExit: If ``GEMINI_API_KEY`` is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Please run: export GEMINI_API_KEY='your_api_key_here'")
        sys.exit(1)
    genai.configure(api_key=api_key)


# ── Scoring ───────────────────────────────────────────────────────────────────

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator for a Visual Question Answering system with \
Explanations (VQA-E). You will be given a Question, the Ground Truth answer \
(with an explanation), and the Model's Predicted answer (also with an \
explanation).

Score the Model's Predicted answer on a scale of 0 to 5 based on how well it \
captures the correct answer AND the core reasoning of the Ground Truth. Ignore \
minor grammatical differences, punctuation, or tokenization artifacts (like \
<unk>). Focus on semantic correctness and reasoning quality.

Question: {question}
Ground Truth: {ground_truth}
Model Prediction: {prediction}

Return ONLY a single integer from 0 to 5. No other text, explanation, or \
markdown.
0: Completely wrong answer and reasoning.
1: Captures a small fragment of the truth but is mostly incorrect.
2: Partially correct but misses key reasoning or details.
3: Mostly correct meaning, but the reasoning is slightly flawed or misses a minor detail.
4: Correct meaning and reasoning, but phrased differently or less comprehensively.
5: Perfect match in meaning and reasoning.\
"""


def _score_with_gemini(
    question: str,
    gt_answer: str,
    pred_answer: str,
    model: genai.GenerativeModel,
) -> Optional[int]:
    """Send one sample to Gemini and parse the returned 0–5 integer score.

    Args:
        question: The VQA question string.
        gt_answer: Ground-truth answer + explanation string.
        pred_answer: Model-predicted answer + explanation string.
        model: Configured ``genai.GenerativeModel`` instance.

    Returns:
        Integer score in ``[0, 5]``, or ``None`` on API failure.
    """
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=gt_answer,
        prediction=pred_answer,
    )
    try:
        response     = model.generate_content(prompt)
        score_text   = response.text.strip()
        # Keep only digit characters — Gemini occasionally adds punctuation.
        score_digits = "".join(filter(str.isdigit, score_text))
        if score_digits:
            score = int(score_digits[0])  # take first digit (response is 0–5)
            return min(max(score, 0), 5)  # clamp to valid range
        return 0
    except Exception as exc:
        print(f"  Gemini API error: {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run LLM-as-a-Judge evaluation and save per-sample scores to JSON."""
    args = _parse_args()
    _setup_gemini()

    model = genai.GenerativeModel(args.model)

    print(f"Loading predictions from {args.input_json} …")
    with open(args.input_json) as f:
        data: list = json.load(f)

    if args.limit is not None:
        data = data[: args.limit]
        print(f"  Limiting to {args.limit} samples.")

    results:      list  = []
    total_score:  int   = 0
    failed_calls: int   = 0

    print(f"Starting LLM-as-a-Judge evaluation using {args.model} …")
    for idx, item in enumerate(tqdm(data)):
        question = item.get("question", "")
        gt       = item.get("ground_truth", "")
        pred     = item.get("prediction", "")

        score = _score_with_gemini(question, gt, pred, model)

        # Single retry with exponential backoff on API failure (e.g. rate limit).
        if score is None:
            failed_calls += 1
            print(f"  Call failed for index {idx}. Retrying in 10 s …")
            time.sleep(10)
            score = _score_with_gemini(question, gt, pred, model)
            if score is None:
                print(f"  Retry failed. Assigning score 0.")
                score = 0

        total_score     += score
        item["llm_score"] = score
        results.append(item)

        # Respect free-tier rate limit (≈15 RPM → 4 s between calls).
        time.sleep(4)

    avg_score = total_score / len(results) if results else 0.0

    print(f"\n--- Evaluation Complete ---")
    print(f"Total samples evaluated : {len(results)}")
    print(f"Failed API calls        : {failed_calls}")
    print(f"Average LLM Score (0–5) : {avg_score:.2f}")

    output = {
        "summary": {
            "total_samples":  len(results),
            "average_score":  avg_score,
            "model_used":     args.model,
            "failed_calls":   failed_calls,
        },
        "results": results,
    }
    print(f"\nSaving detailed results to {args.output_json} …")
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
