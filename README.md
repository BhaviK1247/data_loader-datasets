# data_loader-datasets
configurable sampling layer to balance, filter, and schedule multi-source training corpora for training large language models (LLMs).
Overview

Large corpora used to train LLMs often have wildly different sizes and quality. A Data Sampling Layer sits between raw manifests and the training loop to:

Balance contribution from multiple sources (books, wiki, code, web crawl, curated corpora).

Apply re-weighting (alpha/temperature), curriculum schedules, and stratified constraints.

Offer filtering, deduplication, and monitoring to reduce memorization and bias.

This repo contains:

data_sample_layer.ipynb — main notebook with implementations and experiments.

data_sample_layer-checkpoint.ipynb — checkpointed experiments / plots.

Example config files (in configs/) and small manifests for local testing.

Scripts / snippets to integrate with tokenizers and PyTorch/TF training loops.

Features

Proportional sampling (by dataset size)

Alpha (temperature) reweighting: p_i ∝ n_i^α

Curriculum scheduling: vary sampling weights across epochs/steps

Deduplication hooks (hash/minhash; configurable thresholds)

Filters: token length, regex blacklist, quality score cutoffs

Diagnostics: per-source sample counts, duplicate-rate graphs, coverage reports

Minimal Sampler class ready for adaptation to DataLoader pipelines

Getting started

Requirements (suggested)

python >= 3.9
pip install -r requirements.txt
# typical libs: numpy pandas tqdm scikit-learn datasets matplotlib


Open the notebook:

jupyter lab data_sample_layer.ipynb


Or run the simple example script:

python examples/run_sampler_demo.py --config configs/alpha_0.3.yaml

Usage examples
Compute alpha-weighted probabilities
import numpy as np

sources = ['wiki', 'books', 'commoncrawl']
sizes = np.array([10_000_000, 2_000_000, 200_000_000])
alpha = 0.3
weights = sizes ** alpha
probs = weights / weights.sum()
print(dict(zip(sources, probs.round(4))))

Minimal sampler class (concept)
import random

class SimpleSampler:
    def __init__(self, manifests, probs):
        self.manifests = manifests
        self.probs = probs

    def sample_batch(self, batch_size):
        chosen = random.choices(list(self.probs.keys()), weights=list(self.probs.values()), k=batch_size)
        batch = []
        for s in chosen:
            batch.append(random.choice(self.manifests[s]))
        return batch

Configuration

Example config keys (YAML/JSON):

sources: list of {name, path, size, quality_score}

strategy: proportional | alpha | curriculum | stratified

alpha: float (for alpha reweighting)

curriculum: {start_epoch, end_epoch, start_alpha, end_alpha, schedule}

filters: {min_tokens, max_tokens, blacklist_patterns}

dedupe: {enabled, method, threshold}

batch_size, epoch_length, seed

Place example configs under configs/ and load them in notebooks/scripts.

Integration with training loop

Pseudo-code:

sampler = SimpleSampler(manifests, probs)
for epoch in range(num_epochs):
    if curriculum:
        alpha = curriculum_fn(epoch)
        probs = compute_probs_from_alpha(sizes, alpha)
        sampler.update_probs(probs)
    for step in range(steps_per_epoch):
        batch = sampler.sample_batch(batch_size)
        tokens = tokenizer(batch)
        outputs = model(tokens)
        # backward/update...

Diagnostics & visualizations

Per-source probability bar charts

Per-epoch stacked area for sampled counts

Duplicate rate time series

Token-length histograms by source

Run notebook cells to generate interactive plots.

For GitHub (repo setup & recommended files)

README.md (this file)

LICENSE (e.g., MIT)

.gitignore (ignore venv, data/ large files)

requirements.txt

notebooks/ — place data_sample_layer.ipynb and checkpoint notebook here

configs/ — example YAML configs

examples/ — small runnable demo scripts

data/ — sample manifests (not for large corpora; use manifests that reference remote storage)

docs/ — optional documentation and diagrams

Sample .gitignore lines:

.env
.venv/
__pycache__/
data/
*.ipynb_checkpoints


Git commands to initialize & push:

git init
git add .
git commit -m "Initial commit: data sampling layer for LLMs"
git branch -M main
git remote add origin git@github.com:yourusername/llm-data-sampler.git
git push -u origin main

Contributing

Open issues for bugs/feature requests.

Use PRs on main with clear descriptions and tests where applicable.

Please include reproducible steps for any reported issue.

License

Add your preferred license file (MIT recommended for experimentation/public sharing). Place LICENSE in repo root.

References
Primary reference (requested)

Build LLM From Scratch — treat this as the primary/target reference text for the repository.

If you have the full bibliographic details (author(s), year, publisher, ISBN/URL), replace the placeholder below with that canonical citation.

Suggested placeholder citation format (fill in details):
Author(s). Build LLM From Scratch. Publisher, Year. ISBN/DOI/URL.

How to include the book in the repo:

Add a references/ directory with:

build_llm_from_scratch.md — short notes / key chapters you used.

bibtex.bib — BibTeX entry for the book (if you use academic workflows).

Add a short CITATION.cff file at repo root if you intend others to cite your code.

Additional useful books & resources

Ian Goodfellow, Yoshua Bengio, Aaron Courville — Deep Learning.

Aurélien Géron — Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.

Lewis, Liu, et al. — Natural Language Processing with Transformers.

Research papers and blog posts on dataset curation, weighting, and deduplication (add specific citations in references/ if/when you collect them).
