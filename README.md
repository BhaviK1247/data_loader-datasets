# Data Sampling Layer for LLMs

A lightweight **data sampling layer** for training large language models. It balances multiple datasets (e.g., wiki, books, web) using strategies like proportional sampling, temperature (alpha) reweighting, and curriculum schedules.

---

## Features

* Proportional and temperature-based sampling (`p ∝ n^α`)
* Curriculum scheduling (change mix over epochs)
* Filtering & deduplication hooks
* Simple sampler class + notebooks with examples
* Diagnostic plots (per-source distribution, duplicates, token lengths)

---

## Quick start

```bash
pip install -r requirements.txt
jupyter lab notebooks/data_sample_layer.ipynb
```

Example:

```python
from sampler import SimpleSampler
sampler = SimpleSampler(manifests, probs)
batch = sampler.sample_batch(32)
```

---

## Repo structure

* `notebooks/` — main + checkpoint notebooks
* `configs/` — example YAML configs
* `examples/` — small demo scripts
* `README.md`, `LICENSE`, `requirements.txt`

---

## Reference

📖 **Sebastian Raschka (2024). *Build a Large Language Model (From Scratch)*. Manning Publications.**

Other helpful: *Deep Learning* (Goodfellow et al.), *NLP with Transformers* (Lewis et al.).

---

## License

MIT

---


