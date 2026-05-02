# OpenMythos — Personal HOWTO

> Notes for future-Pedja. Repo lives in this folder. Last updated 2026-05-02.
> Upstream: https://github.com/kyegomez/OpenMythos

---

## What this thing is, in one breath

A PyTorch reconstruction of the **Claude Mythos** architecture: a Recurrent-Depth Transformer (Prelude → Looped Recurrent Block → Coda) with switchable MLA/GQA attention and an MoE feed-forward. Not affiliated with Anthropic — it's research-grade theory built from public papers.

---

## Activate the venv (do this every session)

```bash
cd /Users/dev/work/mythos
source .venv/bin/activate
```

Sanity check that you're in it:
```bash
which python   # should point inside /Users/dev/work/mythos/.venv/bin/
```

When done:
```bash
deactivate
```

---

## First-time install (only if the .venv is missing or broken)

```bash
cd /Users/dev/work/mythos
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                  # editable install from pyproject.toml
# Optional: Flash Attention 2 for GQA (needs CUDA + build tools)
pip install -e ".[flash]"
```

---

## Smoke test — does it work?

There's an `example.py` at the repo root. Run it to confirm the model builds, forwards, and generates:

```bash
python example.py
```

Expected output: parameter count, logits shape `(2, 16, vocab_size)`, generated shape `(2, 24)`, and a spectral radius print line. If that runs without error, you're good.

---

## Try the model variants

Pre-configured scales from 1B → 1T params live in `open_mythos/variants.py`:

```python
from open_mythos import mythos_1b, OpenMythos

cfg = mythos_1b()         # also: mythos_3b, mythos_10b, mythos_50b, mythos_100b, mythos_500b, mythos_1t
model = OpenMythos(cfg)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
```

A working example is at `examples/variants_example.py`. The MoDA (Mixture-of-Depths) example is at `examples/moda_example.py`.

---

## Run the tests

```bash
python -m pytest tests/                       # all tests
python tests/small_benchmark.py               # quick perf sanity check
python tests/bench_vs_transformer.py          # compare looped vs vanilla
```

---

## Train something

The training script for the 3B model on FineWeb-Edu:

```bash
# Single GPU
python training/3b_fine_web_edu.py

# Multi-GPU (auto-detect count)
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
  training/3b_fine_web_edu.py
```

Defaults: AdamW, FineWeb-Edu `sample-10BT`, gpt-oss-20b tokenizer, bf16 on H100/A100, linear-warmup → cosine decay, 30B token target. Edit the script to swap dataset shards or model size.

Heads up: the 3B run is not casual. On a single consumer GPU it'll take... a while. Use the 1B variant for development.

---

## Where to look in the repo

| Path | What's there |
|---|---|
| `open_mythos/main.py` | `OpenMythos` class, `MythosConfig`, the actual architecture |
| `open_mythos/variants.py` | Pre-configured `mythos_1b` ... `mythos_1t` factory functions |
| `open_mythos/moda.py` | Mixture-of-Depths Attention variant |
| `open_mythos/tokenizer.py` | `MythosTokenizer` wrapper around gpt-oss-20b |
| `example.py` | Minimal working example — start here |
| `examples/` | Variant + MoDA examples |
| `tests/` | pytest suite + benchmarks |
| `training/3b_fine_web_edu.py` | Reference training run |
| `docs/open_mythos.md` | Full API reference for the model |
| `docs/datasets.md` | Token-budget guidance per model size |

---

## Mental model — what makes Mythos different

1. **Looped depth, not stacked depth.** The Recurrent Block runs the same weights up to `max_loop_iters` times. More loops at inference = deeper reasoning, no extra parameters.
2. **Reasoning happens in latent space.** Each loop is implicit chain-of-thought — no token output between steps. T loops ≈ T steps of CoT inside one forward pass.
3. **Stability is not free.** The injection matrix `A` must have spectral radius `ρ(A) < 1`, otherwise the hidden state diverges. The repo enforces this via `A := Diag(-exp(log_A))` (Parcae trick).
4. **Breadth comes from MoE.** Fine-grained experts + always-on shared experts in the Recurrent Block FFN. Different experts can fire on different loop iterations — every loop is computationally distinct despite shared weights.
5. **Halting matters.** Naively running max loops on every input causes "overthinking" — the hidden state drifts past the answer. ACT-style adaptive halting is the planned fix.

The full hypothesis is laid out in `README.md` (the project one). Read `docs/open_mythos.md` before modifying any architecture code.

---

## Pitfalls to avoid

1. **Don't run `python -m venv path/to/venv` literally.** Substitute a real path like `.venv`. (Filed under: lessons learned the hard way.)
2. **Always activate the venv before `pip install` or `python`.** If `which python` doesn't point inside `.venv/`, you're polluting system Python.
3. **Check `torch.cuda.is_available()` before training.** Silent CPU fallback wastes hours.
4. **Don't pull `main` blindly.** This repo moves; check the changelog before `git pull` if you've made local edits.
5. **The model is theoretical.** Treat its behavior as research, not production. The hypotheses about Mythos are well-argued speculation, not leaked Anthropic IP.

---

## Quick command reference

```bash
# Activate
source /Users/dev/work/mythos/.venv/bin/activate

# Smoke test
python /Users/dev/work/mythos/example.py

# Build a model interactively
python -c "from open_mythos import mythos_1b, OpenMythos; m = OpenMythos(mythos_1b()); print(sum(p.numel() for p in m.parameters()))"

# Run tests
cd /Users/dev/work/mythos && python -m pytest tests/

# Pull upstream changes
cd /Users/dev/work/mythos && git pull origin main
```
