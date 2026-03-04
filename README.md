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

#### Evaluation (many angles)
- **Ranking quality**: AUC, MRR, nDCG@K, MAP@K, Recall@K
- **Calibration**: ECE (expected calibration error), Brier score
- **Diversity**: intra-list diversity (ILD), category/entity coverage@K, entropy@K
- **Exposure fairness**: position-weighted exposure, disparity vs target distribution (KL / L1 / Gini), new-item exposure floor

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

### 3.4 Evaluate ranker + reranker (metrics + slices)
```bash
python -m mindrec.cli evaluate --config configs/mind_small.yaml
python -m mindrec.cli rerank_eval --config configs/mind_small.yaml
```

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
If you have explicit publisher/source metadata, you can plug it in as provider id.
