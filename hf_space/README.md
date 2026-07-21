---
title: "D&A: Decoupling Description and Assessment"
emoji: 🧬
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: "3.10"
pinned: false
license: mit
---

<h1 align="center">🧬 D&amp;A: Decoupling Description and Assessment<br/>Enables VLMs as Zero-Shot Robotics Reward Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/0000.00000"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"/></a>
  &nbsp;
  <a href="https://github.com/your-org/dna"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
</p>

> **TL;DR** — Naïvely prompting a frontier VLM for robot task progress collapses into a
> success/failure detector that hallucinates progress on failed runs. **D&A** fixes this by
> *decoupling* the job into two prompt stages: first an objective, task-grounded **description**
> of scene dynamics, then an **assessment** against an explicit success rubric. Anchoring the
> score to a global criterion (instead of the video's own timeline) yields calibrated,
> fine-grained progress in `[0, 1]` that penalizes failures and interpolates successes — purely
> prompt-based, so it generalizes across tasks, embodiments, and benchmarks and matches or beats
> post-trained reward models. This Space is an interactive demo.

## What you get

- **Inputs:** an example clip (or your own upload), a task/instruction, a model, and the
  method / thinking level / frame count / sample count (plus an API key if not using the
  Space secret).
- **Outputs:**
  - An annotated **video** — the input clip on top, an animated per-frame progress plot below.
  - **All the model text** — grounded objects, per-frame scene descriptions, and (with
    `D&A (feedback)`) failure/feedback — plus the raw per-frame progress numbers.

## Models & keys

Routing is by model-id prefix (see the `Model id` dropdown — you can type a custom slug):

| Prefix        | Backend            | Key to paste                    |
|---------------|--------------------|---------------------------------|
| `openrouter/` | OpenRouter         | OpenRouter key (`sk-or-…`)      |
| `gemini`      | Google Gemini      | Google AI Studio key            |
| `gpt` / `o*`  | OpenAI             | OpenAI key                      |
| `claude`      | Anthropic          | Anthropic key                   |
| `muse`        | Meta Muse Spark    | Meta key                        |

Paste the key into the **API key** field; it is used only for that request and never stored.

**Zero-config OpenRouter:** OpenRouter also reads the `OPENROUTER_API_KEY` environment variable,
so setting that as a **Space secret** enables `openrouter/*` models without anyone typing a key.
The other backends read only from `dna/secrets.py`, so to preconfigure them fill in the matching
list there (keep the file free of real keys in a public Space) or use the UI field.

## Methods

| Method (UI label)  | Pipeline                              | API calls | Text produced                          |
|--------------------|---------------------------------------|-----------|----------------------------------------|
| `D&A`              | grounding → description → progress    | 3         | objects + per-frame descriptions       |
| `D&A (feedback)`   | like `D&A`, adds failure/feedback     | 3         | + failure & feedback                   |
| `Naive`            | direct video → progress               | 1         | none (progress only)                   |

`n_frames` works best in **8–20** (frame-decomposing backends send one image per frame).
`gemini-2.5*` models require `thinking = HIGH`. Raw model chain-of-thought is **not** surfaced —
the pipeline runs each provider call with `include_thoughts=False`.

## Run locally

```bash
pip install -r requirements.txt
python app.py           # opens the Gradio UI
```

## How it works

The app calls `DNA._sample_once(...)` directly (not the public `compute_progress`, which returns
only the averaged progress array and discards the model's text). It runs `n_samples` pipelines
concurrently, averages the per-frame progress, and renders both the annotated video and the text.
A UI-typed key is injected by temporarily swapping the routed provider's key pool for the duration
of one request and restoring it afterwards, with the swap+run window serialized by a lock so keys
never leak across concurrent users on a shared Space; the vendored `dna` package is otherwise
unmodified from upstream.

## Abstract

Vision-Language Models (VLMs) hold promise for zero-shot video task-progress estimation in robot
learning, yet existing frameworks frequently suffer from confirmation bias, hallucinating progress
in failed executions due to implicit success assumptions or relative temporal ranking designs.
Naïve prompting of frontier models collapses into a binary success detector that fails to capture
nuanced, non-linear progress and failures. While post-training approaches can mitigate this through
explicit training, they demand expensive real-world robotics data and require computationally
intensive post-training for every individual base model. To resolve these limitations, we introduce
D&A (Describe & Assess): our approach decouples progress estimation into an objective,
task-grounded textual description of scene dynamics followed by an assessment phase using explicit
rubrics. By compelling the VLM to synthesize an explicit, text-based definition of absolute success
before scoring, the framework anchors evaluations to an external, global criterion rather than the
video's internal timeline. Empirical results demonstrate that our method calibrates zero-shot
estimates and mitigates confirmation bias, providing fine-grained, partial progress metrics that
accurately penalize failed executions while correctly interpolating successful runs. Being purely
prompt-based, D&A generalizes across tasks, embodiments, and benchmarks — matching or surpassing
post-trained reward models — and its reward estimates enable efficient downstream reinforcement
learning.
