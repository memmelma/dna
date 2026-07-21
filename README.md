# reward-vlm

Vision-Language Model based reward functions for robotic manipulation.

`reward-vlm` turns a video of a robot performing a task, plus a natural-language
task description, into a **per-frame progress signal** in `[0, 1]`
(0 = not started, 1 = complete). It is designed as a plug-and-play reward signal
for robot learning, and works with any of several VLM backends (Gemini, GPT,
Claude, and more) behind a single interface.

## Installation

We recommend [`uv`](https://docs.astral.sh/uv/) for a fast, reproducible setup.
`uv sync` creates the virtual environment and installs the locked dependencies
in one step:

```bash
uv sync                       # base install (all hosted API backends)
uv sync --group qwen          # also install local Qwen-VL inference — in development

# run anything in the environment without activating it:
uv run python -c "from dna import DNA"
# ...or activate it:
source .venv/bin/activate
```

Or with plain `pip`:

```bash
pip install -e .
pip install --group qwen      # optional: local Qwen-VL inference — in development (pip >= 25.1)
```

Copy `dna/secrets.py.example` to `dna/secrets.py` and fill in the API keys for
the providers you plan to use. OpenRouter reads the `OPENROUTER_API_KEY`
environment variable (falling back to `OPENROUTER_API_KEYS` in `secrets.py`).

## Quick start

```python
import numpy as np
from dna import DNA

frames = np.load("rollout.npy")          # (N, H, W, 3) uint8 video
dna = DNA(model="gemini-3-flash-preview", method="dna")
progress = dna.compute_progress(frames, "pick up the red block")
# -> np.ndarray of shape (N,) with values in [0, 1]
```

Inside an existing event loop (e.g. a Jupyter notebook or an async pipeline),
use the async variant instead:

```python
progress = await dna.compute_progress_async(frames, task)
```

## Methods

`DNA` exposes three methods that trade accuracy for cost:

| `method`      | Pipeline                              | API calls |
|---------------|---------------------------------------|-----------|
| `"dna"`       | grounding → description → progress    | 3         |
| `"decompose"` | description → progress                | 2         |
| `"naive"`     | direct video → progress               | 1         |

- **`dna`** first grounds the task-relevant objects, then produces per-frame
  scene descriptions conditioned on those objects, then scores progress from the
  descriptions. Most accurate.
- **`decompose`** skips grounding: it describes each frame, then scores progress
  from the descriptions.
- **`naive`** sends the video directly and asks for per-frame progress in a
  single call. Fastest.

## Supported models

Select a backend via the `model` argument; routing is by prefix.

| Backend                | Example `model`                          | Keys / auth                      |
|------------------------|------------------------------------------|----------------------------------|
| Google Gemini          | `gemini-3-flash-preview`                 | `GOOGLE_API_KEYS`                |
| OpenAI GPT             | `gpt-4o`, `o3`                           | `OPENAI_API_KEYS`                |
| Anthropic Claude       | `claude-sonnet-4-20250514`               | `ANTHROPIC_API_KEYS`             |
| OpenRouter (any model) | `openrouter/openai/gpt-5.6-terra`, `openrouter/openai/gpt-5.6-luna`, `openrouter/anthropic/claude-opus-4.8`, `openrouter/google/gemini-3-pro-preview` | `OPENROUTER_API_KEY` |
| Meta Muse Spark        | `muse-spark-1.1`                         | `META_API_KEYS`                  |
| Qwen-VL (local) ⚠️      | `Qwen/Qwen3-VL-8B-Instruct`              | none (local GPU)                 |

⚠️ **Qwen-VL (local inference) is in development** and not yet fully validated —
use the hosted API backends for production.

Gemini and Muse accept native video; the other backends decompose the video into
per-frame images. **OpenRouter** is a convenient single endpoint for reaching
GPT, Claude, and many other models — especially useful for debugging without
each vendor's own key. Prefix any OpenRouter model slug with `openrouter/`.

## Reasoning effort

The `thinking` argument (`"LOW"`, `"MEDIUM"`, `"HIGH"`, or `"OFF"`) controls the
model's reasoning budget where supported. It maps to Gemini's thinking level,
OpenAI's `reasoning_effort` (reasoning models only), and Claude's extended
thinking budget.

## Advanced

The `Experimental` class (aliased `RVLM` for backward compatibility) exposes the
full research surface — many additional input modalities (stacked frames, single
frames, all-frames grounding, …), processing strategies, and reasoning variants
via a compound `modality` string. `DNA` is the recommended, minimal interface for
most uses.
