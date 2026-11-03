# 🧬 D&A: Decoupling Description and Assessment<br>Enables VLMs as Zero-Shot Robotic Reward Models

[Marius Memmel](https://memmelma.github.io/)<sup>1,2</sup>, [Shawn Yin](https://xheawn.github.io/)<sup>1,\*</sup>, [Urvi Rutia](https://www.linkedin.com/in/urvi-rutia)<sup>1,\*</sup>, [Angad Wadhwa](https://www.linkedin.com/in/angadw/)<sup>1,\*</sup>, Eldon Li<sup>1</sup>, [Gavin Angeloff](https://www.linkedin.com/in/gavin-angeloff-3164452ba/)<sup>1</sup>,
[Ankit Goyal](https://imankgoyal.github.io/)<sup>2</sup>, [Jesse Zhang](https://jessezhang.net/)<sup>1</sup>, [Dieter Fox](https://homes.cs.washington.edu/~fox/)<sup>1</sup>, [Fabio Ramos](https://fabioramos.github.io/Home.html)<sup>2</sup>, [Anqi Li](https://anqili.github.io/)<sup>2,†</sup>, [Abhishek Gupta](https://homes.cs.washington.edu/~abhgupta/)<sup>1,†</sup>

<sup>1</sup>University of Washington, <sup>2</sup>NVIDIA &nbsp;·&nbsp; <sup>\*</sup>equal contribution, <sup>†</sup>equal advising

<p>
  <a href="https://weirdlabuw.github.io/dna/"><b>Website</b></a> &nbsp;·&nbsp;
  <a href="https://weirdlabuw.github.io/dna/demo/"><b>Demo</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/memmelma/dnabench"><b>D&amp;A-Bench dataset</b></a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2611.03622"><b>Paper</b></a>
</p>

**D&A** turns a robot video and a task description into a **per-frame progress signal** in `[0, 1]`
(0 = not started, 1 = complete), with no training and no robot data. It *describes* what happened first, without
seeing the task, and only then *assesses* progress against explicit success criteria, which avoids the confirmation
bias that makes VLMs hallucinate progress on failed rollouts. It works with several VLM backends (Gemini, GPT,
Claude, and more) behind a single interface, and its rewards drive real-world online and offline RL.

## Quick start

```python
from dna import DNA

dna = DNA(model="openrouter/openai/gpt-6-astra")
progress = dna.compute_progress(video, "assemble the car")
```

`video` is an `(N, H, W, 3)` `uint8` array; `progress` is an `(N,)` array with values in `[0, 1]`
(0 = not started, 1 = complete). Set `OPENROUTER_API_KEY` (see [Installation](#installation)) or pick another
backend from [Supported models](#supported-models).

Inside an existing event loop (e.g. a Jupyter notebook or an async pipeline),
use the async variant instead:

```python
progress = await dna.compute_progress_async(video, task)
```

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

## Citation

If you find D&A useful for your work, please cite:

```bibtex
@misc{memmel2026dna,
  title={D\&A: Decoupling Description and Assessment Enables VLMs as Zero-Shot Robotic Reward Models},
  author={Memmel, Marius and Yin, Shawn and Rutia, Urvi and Wadhwa, Angad and Li, Eldon and Angeloff, Gavin and Goyal, Ankit and Zhang, Jesse and Fox, Dieter and Ramos, Fabio and Li, Anqi and Gupta, Abhishek},
  year={2026},
  eprint={2611.03622},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2611.03622}
}
```

## License

Released under the [MIT License](LICENSE).
