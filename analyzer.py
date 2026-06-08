"""
analyzer.py — FeedbackIQ Core Analysis Engine
================================================
What makes this different from a standard sentiment analyzer:

  ABSA (Aspect-Based Sentiment Analysis)
  ────────────────────────────────────────
  Standard tools return one score: "Positive" or "Negative."
  FeedbackIQ decomposes each review into 5 business-critical dimensions:

    • Product Quality    — Is the core offering meeting expectations?
    • Customer Service   — How is human/support interaction perceived?
    • Delivery           — Speed, packaging, logistics experience
    • Value for Money    — Price-to-quality perception
    • User Experience    — Ease of use, interface, process clarity

  This is a real NLP research area (Pontiki et al., 2014 SemEval ABSA task).
  Knowing *where* sentiment is negative is infinitely more actionable than
  knowing that overall sentiment is negative.

References:
    Pontiki et al. (2014). SemEval-2014 Task 4: Aspect-Based Sentiment Analysis.
    https://aclanthology.org/S14-2004/
"""

import io
import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
ASPECTS = [
    "product_quality",
    "customer_service",
    "delivery",
    "value_for_money",
    "user_experience",
]

ASPECT_LABELS = {
    "product_quality":   "Product Quality",
    "customer_service":  "Customer Service",
    "delivery":          "Delivery",
    "value_for_money":   "Value for Money",
    "user_experience":   "User Experience",
}

_SYSTEM = (
    "You are a senior business intelligence analyst specialising in "
    "customer experience and brand perception. "
    "Analyse the provided feedback and return ONLY a valid JSON object. "
    "No markdown fences, no preamble, no trailing text."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _client() -> Groq | None:
    key = os.environ.get("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


def _clean_json(raw: str) -> str:
    """Strip everything outside the outermost { } or [ ]."""
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    return match.group(0) if match else raw


def _build_prompt(text: str) -> str:
    return f"""Analyse this customer feedback:

\"\"\"{text}\"\"\"

Return a single JSON object with EXACTLY these fields:

{{
  "sentiment": "Positive" | "Negative" | "Neutral" | "Mixed",
  "sentiment_score": <float, -1.0 to 1.0>,
  "emotion": "<primary emotion: Satisfied | Frustrated | Disappointed | Delighted | Indifferent | Angry | Grateful | Confused>",
  "confidence": "High" | "Medium" | "Low",
  "summary": "<1–2 sentence analytical summary for a business manager>",
  "key_topics": ["<topic>", ...],
  "positive_aspects": ["<specific positive point>", ...],
  "negative_aspects": ["<specific issue>", ...],
  "actionable_recommendation": "<one concrete, business-ready action>",
  "priority": "High" | "Medium" | "Low",
  "business_impact": "High" | "Medium" | "Low",
  "aspect_scores": {{
    "product_quality":  <float -1.0 to 1.0, or null if not mentioned>,
    "customer_service": <float -1.0 to 1.0, or null if not mentioned>,
    "delivery":         <float -1.0 to 1.0, or null if not mentioned>,
    "value_for_money":  <float -1.0 to 1.0, or null if not mentioned>,
    "user_experience":  <float -1.0 to 1.0, or null if not mentioned>
  }}
}}

Rules for aspect_scores:
- Score ONLY aspects explicitly or implicitly mentioned in the text.
- Use null for aspects not mentioned (do NOT default to 0).
- Score range: -1.0 = extremely negative, 0.0 = neutral, +1.0 = extremely positive.
"""


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_feedback(feedback_text: str) -> dict:
    """
    Run ABSA on a single customer review.

    Returns:
        {
          "success": bool,
          "data": { ...ABSA fields... },   # present if success=True
          "error": "..."                   # present if success=False
        }
    """
    client = _client()
    if not client:
        return {
            "success": False,
            "error": (
                "GROQ_API_KEY not set. "
                "Add it in HuggingFace Space → Settings → Repository secrets."
            ),
        }

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _build_prompt(feedback_text)},
            ],
            temperature=0.05,   # near-deterministic for consistent business output
            max_tokens=750,
        )
        raw  = resp.choices[0].message.content.strip()
        data = json.loads(_clean_json(raw))

        # Normalise: ensure all aspect keys exist (fill missing with None)
        data.setdefault("aspect_scores", {})
        for a in ASPECTS:
            data["aspect_scores"].setdefault(a, None)

        return {"success": True, "data": data}

    except json.JSONDecodeError:
        return {"success": False, "error": "Model returned non-JSON output. Try again."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def analyze_batch(feedbacks_text: str) -> list[dict]:
    """
    Analyse multiple reviews (one per line).

    Returns a list of result dicts, each with:
        result["original_text"]  — the original review string
        result["index"]          — 1-based index
        ...standard analyze_feedback fields...
    """
    lines = [l.strip() for l in feedbacks_text.splitlines() if len(l.strip()) > 5]
    results = []
    for i, line in enumerate(lines, 1):
        r = analyze_feedback(line)
        r["original_text"] = line
        r["index"] = i
        results.append(r)
    return results


def analyze_csv_file(file_path: str) -> tuple[list[dict], str]:
    """
    Analyse all reviews in a CSV file.

    Auto-detects the review/feedback column by searching for common column
    name patterns. Falls back to the first column if none match.

    Returns:
        (results_list, detected_column_name)
    """
    import pandas as pd

    df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    # ── Auto-detect text column ────────────────────────────────────────────
    keywords = ["review", "feedback", "comment", "text", "description", "body", "content"]
    text_col = next(
        (c for c in df.columns if any(kw in c.lower() for kw in keywords)),
        df.columns[0],
    )

    results = []
    for i, row in df.iterrows():
        text = str(row[text_col]).strip()
        if len(text) < 5 or text.lower() in ("nan", "none", ""):
            continue
        r = analyze_feedback(text)
        r["original_text"] = text
        r["index"] = i + 1
        if r["success"]:
            # Carry along any extra metadata columns (date, product, store, etc.)
            r["metadata"] = {
                col: str(row[col])
                for col in df.columns
                if col != text_col
            }
        results.append(r)

    return results, text_col


# ── Export ────────────────────────────────────────────────────────────────────

def results_to_csv_bytes(results: list[dict]) -> bytes:
    """
    Convert a list of analysis results into a downloadable CSV.
    One row per review; includes all ABSA aspect scores as percentage columns.
    """
    import pandas as pd

    rows = []
    for r in results:
        if not r.get("success"):
            rows.append({
                "Review":    r.get("original_text", ""),
                "Error":     r.get("error", "Analysis failed"),
            })
            continue

        d = r["data"]
        row = {
            "Review":           r.get("original_text", ""),
            "Sentiment":        d.get("sentiment"),
            "Score (-1 to 1)":  d.get("sentiment_score"),
            "Emotion":          d.get("emotion"),
            "Confidence":       d.get("confidence"),
            "Priority":         d.get("priority"),
            "Business Impact":  d.get("business_impact"),
            "Summary":          d.get("summary"),
            "Key Topics":       ", ".join(d.get("key_topics", [])),
            "Positives":        ", ".join(d.get("positive_aspects", [])),
            "Issues":           ", ".join(d.get("negative_aspects", [])),
            "Recommendation":   d.get("actionable_recommendation"),
        }

        # Aspect scores as 0–100 percentage strings
        for asp in ASPECTS:
            score = d.get("aspect_scores", {}).get(asp)
            label = ASPECT_LABELS[asp]
            row[label] = f"{(score + 1) / 2 * 100:.0f}%" if score is not None else "N/A"

        rows.append(row)

    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()