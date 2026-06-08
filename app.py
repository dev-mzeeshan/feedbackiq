"""
app.py - FeedbackIQ: Aspect-Based Customer Intelligence
=========================================================
Three analysis modes:
  1. Single Review  - ABSA breakdown + interactive radar chart
  2. Batch Text     - Multi-review summary + sentiment pie + aspect bar chart
  3. CSV Upload     - Bulk file processing + downloadable Excel report

Run locally:  python app.py  (or: gradio app.py)
Deploy:       HuggingFace Spaces (set GROQ_API_KEY as a Repository secret)
"""

import os
import tempfile

import gradio as gr
import plotly.graph_objects as go

from analyzer import (
    ASPECT_LABELS,
    ASPECTS,
    analyze_batch,
    analyze_csv_file,
    analyze_feedback,
    results_to_csv_bytes,
)

# ── Chart builders ────────────────────────────────────────────────────────────

def _score_to_pct(score: float) -> float:
    """Convert -1…1 score to 0…100 percentage."""
    return (score + 1) / 2 * 100


def make_radar_chart(aspect_scores: dict) -> go.Figure | None:
    """
    Radar / spider chart showing per-aspect sentiment percentages.

    Returns None when no aspects are scored (all null) so the UI
    can hide the chart gracefully.
    """
    valid = {k: v for k, v in aspect_scores.items() if v is not None}
    if not valid:
        return None

    labels  = [ASPECT_LABELS[k] for k in valid]
    values  = [_score_to_pct(v) for v in valid.values()]

    # Close the polygon
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=6, color="#6366f1"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(size=10, color="#94a3b8"),
                gridcolor="rgba(148,163,184,0.15)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                gridcolor="rgba(148,163,184,0.15)",
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=50, t=40, b=40),
        height=340,
    )
    return fig


def make_sentiment_pie(results: list[dict]) -> go.Figure | None:
    """Donut chart - sentiment distribution across a batch."""
    counts: dict[str, int] = {}
    for r in results:
        if r.get("success"):
            s = r["data"].get("sentiment", "Unknown")
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None

    colour_map = {
        "Positive": "#22c55e",
        "Negative": "#ef4444",
        "Neutral":  "#94a3b8",
        "Mixed":    "#f59e0b",
    }
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.45,
        marker_colors=[colour_map.get(k, "#6366f1") for k in counts],
        textfont=dict(size=12),
    ))
    fig.update_layout(
        title=dict(text="Sentiment Distribution", font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        legend=dict(font=dict(size=11)),
        height=300,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def make_aspect_bar(results: list[dict]) -> go.Figure | None:
    """
    Horizontal bar chart - average aspect score across the batch.
    Only includes aspects mentioned in at least one review.
    """
    acc: dict[str, list[float]] = {a: [] for a in ASPECTS}
    for r in results:
        if r.get("success"):
            for a in ASPECTS:
                v = r["data"].get("aspect_scores", {}).get(a)
                if v is not None:
                    acc[a].append(v)

    means = {
        ASPECT_LABELS[a]: sum(vs) / len(vs) * 50 + 50   # -1..1 → 0..100
        for a, vs in acc.items() if vs
    }
    if not means:
        return None

    labels = list(means.keys())
    values = list(means.values())
    bar_colours = ["#22c55e" if v >= 60 else "#f59e0b" if v >= 40 else "#ef4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=bar_colours,
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text="Average Aspect Scores", font=dict(size=14)),
        xaxis=dict(
            range=[0, 115],
            ticksuffix="%",
            gridcolor="rgba(148,163,184,0.1)",
            zeroline=False,
        ),
        yaxis=dict(automargin=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        height=280,
        margin=dict(l=10, r=60, t=45, b=20),
    )
    return fig


# ── Formatters ────────────────────────────────────────────────────────────────

_PRIORITY_LABELS = {
    "High":   "🔴 High - Immediate action needed",
    "Medium": "🟡 Medium - Address this week",
    "Low":    "🟢 Low - Monitor for trends",
}
_IMPACT_LABELS = {
    "High":   "🔴 High",
    "Medium": "🟡 Medium",
    "Low":    "🟢 Low",
}


def _bar(score: float, width: int = 18) -> str:
    """ASCII progress bar for sentiment score (-1…1)."""
    pct    = _score_to_pct(score)
    filled = round(pct / 100 * width)
    return "▓" * filled + "░" * (width - filled) + f"  {pct:.0f}%"


def format_single(result: dict) -> str:
    if not result["success"]:
        return f"### ⚠️ Error\n{result['error']}"

    d       = result["data"]
    score   = d.get("sentiment_score", 0)
    aspects = d.get("aspect_scores", {})

    pos_lines = "\n".join(f"- {p}" for p in d.get("positive_aspects", [])) or "- None identified"
    neg_lines = "\n".join(f"- {n}" for n in d.get("negative_aspects", [])) or "- None identified"

    # Aspect table rows (skip null aspects)
    aspect_rows = "\n".join(
        f"| {ASPECT_LABELS[a]} | {_bar(v, 14)} |"
        for a, v in aspects.items()
        if v is not None
    )
    aspect_section = (
        f"### 📊 Aspect Scores\n| Dimension | Score |\n|:--|:--|\n{aspect_rows}\n"
        if aspect_rows else ""
    )

    return f"""### 📋 Summary
{d.get('summary', '-')}

---

| Field | Value |
|:--|:--|
| **Sentiment** | {d.get('sentiment', '-')} |
| **Emotion** | {d.get('emotion', '-')} |
| **Score** | `{_bar(score)}` |
| **Confidence** | {d.get('confidence', '-')} |
| **Priority** | {_PRIORITY_LABELS.get(d.get('priority'), d.get('priority', '-'))} |
| **Business Impact** | {_IMPACT_LABELS.get(d.get('business_impact'), d.get('business_impact', '-'))} |

{aspect_section}
### ✅ Positive Aspects
{pos_lines}

### ❌ Issues Identified
{neg_lines}

---
### 💡 Recommendation
{d.get('actionable_recommendation', '-')}
"""


def format_batch(results: list[dict]) -> str:
    if not results:
        return "No valid reviews found."

    ok      = [r for r in results if r.get("success")]
    total   = len(results)
    counts  = {}
    for r in ok:
        s = r["data"].get("sentiment", "Unknown")
        counts[s] = counts.get(s, 0) + 1

    header = f"### 📊 Batch Summary - {total} Reviews Analysed\n\n"
    header += "| " + " | ".join(counts.keys()) + " |\n"
    header += "|" + "---|" * len(counts) + "\n"
    header += "| " + " | ".join(str(v) for v in counts.values()) + " |\n\n---\n"

    body = ""
    for r in results:
        preview = r["original_text"][:90] + ("…" if len(r["original_text"]) > 90 else "")
        body += f"\n**#{r['index']}** - {preview}\n"
        body += format_single(r) + "\n---\n"

    return header + body


# ── Gradio handlers ───────────────────────────────────────────────────────────

def run_single(text: str):
    """Handler: single review → (markdown, radar_chart)."""
    if not text or len(text.strip()) < 10:
        return "Please enter at least 10 characters.", None

    result = analyze_feedback(text)
    md     = format_single(result)
    chart  = make_radar_chart(result["data"].get("aspect_scores", {})) if result["success"] else None
    return md, chart


def run_batch(text: str):
    """Handler: batch text → (markdown, pie, bar, csv_path)."""
    if not text or len(text.strip()) < 5:
        return "Please enter at least one review.", None, None, None

    results  = analyze_batch(text)
    md       = format_batch(results)
    pie      = make_sentiment_pie(results)
    bar      = make_aspect_bar(results)
    csv_path = _save_csv(results)
    return md, pie, bar, csv_path


def run_csv(file):
    """Handler: uploaded CSV → (markdown, pie, bar, csv_path)."""
    if file is None:
        return "Please upload a CSV file.", None, None, None
    try:
        results, col = analyze_csv_file(file.name)
        md       = f"*Detected review column: `{col}`*\n\n" + format_batch(results)
        pie      = make_sentiment_pie(results)
        bar      = make_aspect_bar(results)
        csv_path = _save_csv(results)
        return md, pie, bar, csv_path
    except Exception as e:
        return f"### ⚠️ Error\n{e}", None, None, None


def _save_csv(results: list[dict]) -> str:
    """Write analysis results to a temp CSV file and return its path."""
    data = results_to_csv_bytes(results)
    path = os.path.join(tempfile.gettempdir(), "feedbackiq_report.csv")
    with open(path, "wb") as f:
        f.write(data)
    return path


# ── Sample data ───────────────────────────────────────────────────────────────

_POS = (
    "The customer support team was incredibly helpful and resolved my issue within "
    "10 minutes. The product itself exceeded my expectations. Will definitely recommend!"
)
_NEG = (
    "I've been waiting 3 weeks for my order with no update. Customer service is "
    "unreachable and when I finally got through, they couldn't help at all. Very disappointed."
)
_MIX = (
    "The product quality is excellent and works exactly as described. However, "
    "delivery took much longer than expected and the packaging arrived damaged. Mixed feelings."
)
_BATCH = """\
Great product, fast shipping, very satisfied with the purchase!
Terrible experience. The app keeps crashing and support never responds.
Average service, nothing special but gets the job done.
I love the new features but the price increase is unreasonable.
Delivery was late but the product quality made up for it.\
"""

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
:root {
    --accent:     #6366f1;
    --accent-dim: rgba(99,102,241,0.12);
    --border:     rgba(148,163,184,0.15);
    --muted:      #64748b;
    --surface:    #0f172a;
}
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }

/* Result output */
.md-out table  { width:100%!important; border-collapse:collapse!important; margin:12px 0!important; }
.md-out th     { background:var(--accent-dim)!important; padding:10px!important; color:var(--accent)!important; text-align:left!important; }
.md-out td     { padding:10px!important; border-bottom:1px solid var(--border)!important; }
.md-out code   { background:var(--accent-dim)!important; padding:2px 6px!important; border-radius:4px!important; font-family:monospace!important; }

/* Tabs */
.tab-nav button       { font-weight:600!important; }
.tab-nav button.selected { color:var(--accent)!important; border-bottom-color:var(--accent)!important; }

/* Chart area */
.chart-panel { background: rgba(15,23,42,0.6)!important; border-radius:12px!important; padding:8px!important; }
footer { display:none!important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    css=CSS,
    title="FeedbackIQ - Aspect-Based Customer Intelligence",
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("""
# 🔍 FeedbackIQ
### Aspect-Based Customer Intelligence
*Unlike standard sentiment analyzers, FeedbackIQ decomposes each review into
**5 business-critical dimensions** - giving you actionable intelligence, not just a score.*

Built by [**Muhammad Zeeshan**](https://dev-zeeshan-portfolio.vercel.app) &nbsp;·&nbsp;
[GitHub](https://github.com/dev-mzeeshan/feedbackiq) &nbsp;·&nbsp;
[LinkedIn](https://linkedin.com/in/zeeshanofficial)
""")

    # ── Tab 1: Single Review ──────────────────────────────────────────────────
    with gr.Tab("🔎 Single Review"):
        gr.Markdown(
            "<p style='color:#64748b;font-size:13px;margin:0 0 12px'>"
            "Analyse one review. See per-aspect scores on the radar chart - "
            "not just overall sentiment.</p>"
        )

        with gr.Row():
            with gr.Column(scale=2):
                single_in = gr.Textbox(
                    label="Customer Review",
                    placeholder="Paste a customer review here…",
                    lines=5,
                    max_lines=14,
                )
                with gr.Row():
                    analyse_btn = gr.Button("Analyse", variant="primary", scale=3)
                    clear_btn   = gr.Button("Clear",   variant="secondary", scale=1)

                gr.Markdown("<p style='font-size:12px;color:#64748b;margin:10px 0 6px'>Try a sample:</p>")
                with gr.Row():
                    btn_pos = gr.Button("Positive", size="sm")
                    btn_neg = gr.Button("Negative", size="sm")
                    btn_mix = gr.Button("Mixed",    size="sm")

            with gr.Column(scale=3):
                radar_plot = gr.Plot(
                    label="Aspect Sentiment Radar",
                    elem_classes=["chart-panel"],
                )

        single_out = gr.Markdown(
            value="*Analysis will appear here after clicking Analyse.*",
            elem_classes=["md-out"],
        )

        # Events
        analyse_btn.click(run_single, inputs=single_in, outputs=[single_out, radar_plot])
        clear_btn.click(
            lambda: ("", "*Analysis will appear here.*", None),
            outputs=[single_in, single_out, radar_plot],
        )
        btn_pos.click(lambda: _POS, outputs=single_in)
        btn_neg.click(lambda: _NEG, outputs=single_in)
        btn_mix.click(lambda: _MIX, outputs=single_in)

    # ── Tab 2: Batch Text ─────────────────────────────────────────────────────
    with gr.Tab("📦 Batch Analysis"):
        gr.Markdown(
            "<p style='color:#64748b;font-size:13px;margin:0 0 12px'>"
            "Paste multiple reviews (one per line). Get per-review ABSA breakdowns "
            "plus an aggregate dashboard with sentiment distribution and aspect scores.</p>"
        )

        batch_in = gr.Textbox(
            label="Multiple Reviews (one per line)",
            placeholder="Review 1…\nReview 2…\nReview 3…",
            lines=6,
            max_lines=30,
        )
        with gr.Row():
            batch_btn    = gr.Button("Analyse All",  variant="primary",   scale=3)
            sample_btn   = gr.Button("Load Sample",  variant="secondary", scale=1)

        with gr.Row():
            pie_plot = gr.Plot(label="Sentiment Distribution", elem_classes=["chart-panel"])
            bar_plot = gr.Plot(label="Average Aspect Scores",  elem_classes=["chart-panel"])

        batch_out   = gr.Markdown(
            value="*Results will appear here after clicking Analyse All.*",
            elem_classes=["md-out"],
        )
        batch_csv   = gr.File(label="📥 Download Report (CSV)", visible=False)

        # Events
        batch_btn.click(
            run_batch,
            inputs=batch_in,
            outputs=[batch_out, pie_plot, bar_plot, batch_csv],
        )
        batch_btn.click(lambda: gr.File(visible=True), outputs=batch_csv)
        sample_btn.click(lambda: _BATCH, outputs=batch_in)

    # ── Tab 3: CSV Upload ─────────────────────────────────────────────────────
    with gr.Tab("📂 CSV Upload"):
        gr.Markdown(
            "<p style='color:#64748b;font-size:13px;margin:0 0 12px'>"
            "Upload a CSV file containing customer reviews. FeedbackIQ auto-detects "
            "the review column, analyses every row, and returns a downloadable report "
            "with all ABSA scores per review.</p>"
        )

        csv_upload = gr.File(
            label="Upload CSV (review column auto-detected)",
            file_types=[".csv"],
        )
        csv_btn = gr.Button("Process File", variant="primary")

        with gr.Row():
            csv_pie = gr.Plot(label="Sentiment Distribution", elem_classes=["chart-panel"])
            csv_bar = gr.Plot(label="Average Aspect Scores",  elem_classes=["chart-panel"])

        csv_out  = gr.Markdown(
            value="*Upload a CSV file and click Process File.*",
            elem_classes=["md-out"],
        )
        csv_dl   = gr.File(label="📥 Download Full Report (CSV)", visible=False)

        csv_btn.click(run_csv, inputs=csv_upload, outputs=[csv_out, csv_pie, csv_bar, csv_dl])
        csv_btn.click(lambda: gr.File(visible=True), outputs=csv_dl)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown("""
<p style='font-size:12px;color:#475569;text-align:center;margin-top:24px'>
Powered by Groq API &nbsp;·&nbsp; Llama 3.3 70B &nbsp;·&nbsp; Gradio 5 &nbsp;·&nbsp;
ABSA methodology: Pontiki et al. (2014) SemEval &nbsp;·&nbsp;
<a href='https://github.com/dev-mzeeshan/feedbackiq'>View on GitHub</a>
</p>
""")


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False)