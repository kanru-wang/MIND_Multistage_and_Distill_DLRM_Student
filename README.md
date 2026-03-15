# MIND Multi-Stage Recommender (Retrieval → DLRM Ranker → Diversity+Fairness Re-ranker)

This project implements a realistic recommender stack on the **Microsoft News Dataset (MIND)**:
- Stage 1: **Teacher retrieval** (two-tower, text-based item encoder + history-based user encoder)
- Stage 2: **Student ranker** (DLRM-style sparse+dense model) with **attention fusion** to ingest teacher embeddings
- Stage 3: **Re-ranking** enforcing **(1) relevance vs novelty** and **(2) coverage of categories/entities**, plus **exposure fairness** constraints/penalties for category/provider/new items
- Extensive **evaluation**: ranking metrics, calibration, diversity, exposure fairness, and cold/new slices

MIND is widely used as a benchmark for news recommendation, with impression logs and rich news metadata.

#### Recommender architecture
- Candidate generation with ANN search using **Faiss** (CPU-friendly).
- A **DLRM-style** ranker (dense MLP + sparse embeddings + feature interaction).
- Knowledge **distillation** (logit + representation) from a powerful teacher into a cheaper student (better cold/new performance vs training the student from scratch).

#### Teacher model (simple view)
- The teacher is a pair of encoders:
- a **news/item encoder** that maps each article into an embedding vector
- a **user encoder** that maps a user’s recent clicked history into an embedding vector
- Training goal (standard retrieval idea): make clicked `(user, news)` pairs close in embedding space, and non-clicked pairs farther apart (contrastive learning with in-batch/sampled negatives).

#### Evaluation (many angles)
- **Ranking quality**: AUC, MRR, nDCG@K, MAP@K, Recall@K
- **Calibration**: ECE (expected calibration error), Brier score
- **Diversity**: intra-list diversity (ILD), category/entity coverage@K, entropy@K
- **Exposure fairness**: position-weighted exposure, disparity vs target distribution (KL / L1 / Gini), new-item exposure floor

Fairness target note:
- In the current reranker/evaluation code, `fairness.category_target: "catalog"` means the category distribution of the impression candidate pool (the candidates available for that user/impression), not the selected top-K list and not the global corpus-wide catalog.

#### Re-ranking (important implementation note)
- In this project, re-ranking is a **deterministic optimization layer** on top of ranker scores.
- It is controlled by hyperparameters/constraints (relevance, novelty, coverage, fairness).
- **No training loop is required** for this re-ranking stage.

---

## 0) Hardware target

This repo is designed to run on a powerful Windows laptop:
- uses **faiss-cpu**
- uses a small, strong sentence-transformer as teacher by default (MiniLM family), and caches item embeddings
- supports sub-sampling MIND-large, or using **MIND-small** first

---

## 1) Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 2) Get the dataset (MIND)

Place files under:
```
data/raw/MINDsmall_train/
data/raw/MINDsmall_dev/
```
Each folder should contain `behaviors.tsv` and `news.tsv`.

MIND also provides `entity_embedding.vec` and `relation_embedding.vec`.
In this repo, for simplicity, these two files are currently **not used** by the pipeline.

---

## 2.1) Quick terminology: entities in MIND

- In MIND, an **entity** is a named entity extracted from a news article (person, organization, location, etc.) and linked to a knowledge graph (the MIND paper references Wikidata).
- `entity_embedding.vec`: embedding vector for each entity ID.
- `relation_embedding.vec`: embedding vector for each relation type between entities.

If you choose to use these files, a common approach is to use KG triples `(entity, relation, entity)` to fetch neighbors of entities mentioned in a news article, then build richer representations (for example with graph attention or memory-network style modules).

---

## 2.2) What `run_preprocess()` does (plain English)

`run_preprocess()` converts raw MIND TSV files into model-ready parquet/json files.

Main steps:
- Read `news.tsv` and `behaviors.tsv` from train/dev.
- Build ID mappings (`user_id/news_id/category/subcategory -> integer index`).
- Build pairwise training rows for ranker training (`train_pairs.parquet` and `dev_pairs.parquet`).
- Build impression-level dev data for evaluation (`dev_impressions.parquet`).

How pairs are created:
- For an impression with `P` positives and `N` negatives, this code contributes:
- `P * (1 + min(4, N))` pairs
- because each positive is paired with up to 4 sampled negatives.

Why there is no `train_impressions.parquet`:
- Training uses pairwise rows (`train_pairs.parquet`), not full impression-grouped rows.
- Impression-grouped data is mainly needed for ranking evaluation, so only `dev_impressions.parquet` is generated.

---

## 3) Quickstart (end-to-end on a small slice)

### 3.1 Preprocess TSV → Parquet + feature maps
```bash
python -m mindrec.cli preprocess --config configs/mind_small.yaml
```

### 3.2 Train teacher retriever + build ANN index
```bash
python -m mindrec.cli train_teacher --config configs/mind_small.yaml
python -m mindrec.cli build_index --config configs/mind_small.yaml
python -m mindrec.cli eval_retrieval --config configs/mind_small.yaml
```

### 3.3 Train student DLRM ranker with distillation
```bash
python -m mindrec.cli train_ranker --config configs/mind_small.yaml
```

`train_ranker` also fits a post-hoc temperature scaler on held-out `dev_pairs` and saves it to `runs/<run_name>/ranker/calibration.json`. It tunes a single positive scalar `T` in `sigmoid(logit / T)` against held-out labels, improving probability calibration without changing ranking order.

### 3.4 Evaluate ranker + reranker (metrics + slices)
```bash
python -m mindrec.cli evaluate --config configs/mind_small.yaml
python -m mindrec.cli rerank_eval --config configs/mind_small.yaml
```

### 3.5 Search reranker hyperparameters under a product constraint
```bash
python -m mindrec.cli rerank_search --config configs/mind_small.yaml
```

The current reranker defaults in `configs/mind_small.yaml` were selected with a pragmatic product constraint on dev:
- `nDCG@10` drop must be at most `1.0%` vs the relevance-only top-10 baseline.
- `new_item_exposure_frac` must improve by at least `+0.015`.
- `category_coverage@10` must improve by at least `+0.30`.
- `fairness_kl` may worsen by at most `+0.002`.

The current selected setting is:
- `relevance_weight=0.90`
- `novelty_weight=0.05`
- `coverage_weight=0.05`
- `novelty_sim=teacher_cosine`
- `fairness.penalty_weight=0.5`
- `fairness.new_item_floor=0.20`

The search writes its summary to `runs/<run_name>/eval/rerank_search.json`.

Artifacts go to `runs/<run_name>/`.

---

## 4) Repo layout

- `src/mindrec/`
  - `data/`: parsing + feature building
  - `models/teacher_*`: teacher two-tower retrieval
  - `models/dlrm_*`: student DLRM ranker + attention fusion
  - `rerank/`: diversity + coverage + exposure fairness reranking
  - `metrics/`: ranking, calibration, diversity, fairness
  - `cli.py`: entrypoint for scripts
- `configs/`: YAML configs

---

## 5) Notes on “provider fairness”
MIND includes strong **category/subcategory** metadata. For “provider” fairness, this repo supports:
- **category/subcategory** as providers (default), and/or
- **entity clusters** as proxy providers (optional)
