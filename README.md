# MIND Multi-Stage Recommender (Retrieval → DLRM Ranker → Diversity+Fairness Re-ranker)

This project implements a realistic recommender stack on the **Microsoft News Dataset (MIND)**:
- Stage 1: **Teacher retrieval embeddings** (text-based item encoder + history-based user encoder)
- Stage 2: **Student ranker** (DLRM-style sparse+dense model) with **attention fusion** to ingest teacher embeddings
- Stage 3: **Re-ranking** enforcing **(1) relevance vs novelty** and **(2) category/entity-informed coverage bonuses**, plus **exposure fairness** constraints/penalties for category and new items
- Extensive **evaluation**: ranking metrics, calibration, diversity, exposure fairness, and cold/new slices

MIND is widely used as a benchmark for news recommendation, with impression logs and rich news metadata.

#### Recommender architecture
- Candidate generation with ANN search using **Faiss** (CPU-friendly).
- A **DLRM-style** ranker (dense MLP + sparse embeddings + feature interaction).
- Knowledge **distillation** (logit + representation) from a powerful teacher into a cheaper student (better cold/new performance vs training the student from scratch).

#### Teacher model (simple view)
- The teacher is a learned two-tower retrieval model:
- a **news/item encoder** that starts from frozen sentence-transformer news embeddings and applies a trainable projection
- a **user encoder** that attention-pools clicked-history item embeddings into a user vector
- Training uses clicked positives plus in-impression negatives.
- Retrieval evaluation encodes each dev impression with that impression's own history, so the query is temporally aligned with the impression being scored.

#### Evaluation (many angles)
- **Ranking quality**: AUC, MRR, nDCG@K, MAP@K, Recall@K
- **Calibration**: ECE (expected calibration error), Brier score
- **Diversity**: intra-list diversity (ILD), category coverage@K, category entropy@K
- **Exposure fairness**: position-weighted exposure, disparity vs target distribution (KL / L1 / Gini), new-item exposure floor

Fairness target note:
- In the current reranker/evaluation code, `fairness.category_target: "catalog"` means the category distribution of the impression candidate pool (the candidates available for that user/impression), not the selected top-K list and not the global corpus-wide catalog.
- `rerank_eval` and `rerank_search` now report both `fairness_kl_pool` and `fairness_kl_full`.
- `fairness_kl_pool` compares top-K exposure against the reranker's top-`pool_size` candidate mix.
- `fairness_kl_full` compares top-K exposure against the full impression candidate set.
- Product constraints in reranker search use `fairness_kl_pool`, because that matches the reranker's actual optimization target.

#### Re-ranking (important implementation note)
- In this project, re-ranking is a **deterministic optimization layer** on top of ranker scores.
- It is controlled by hyperparameters/constraints (relevance, novelty, coverage, fairness).
- **No training loop is required** for this re-ranking stage.

#### Distillation representation note
- In `DLRMStudent.forward()`, the student representation used for distillation is `rep = [eu, ei, zf]`, where:
- `eu`: user ID embedding
- `ei`: news/item ID embedding
- `zf`: teacher-guided fusion vector from attention over teacher user/item embeddings
- The dense tower output `xd` and the category/subcategory embeddings `ec`, `es` are intentionally excluded from `rep`.
- Reason: the teacher representation being matched is `concat(teacher_user_emb, teacher_item_emb)`, which is a semantic user-item representation. `eu`, `ei`, and `zf` are the student parts most closely aligned with that semantic space, while `xd`, `ec`, and `es` do not have a clean one-to-one counterpart in the teacher embedding space.

#### Re-ranking process
- `greedy_rerank()` takes the top `pool_size` candidates by ranker score, then builds the final top-`k_out` list one item at a time.
- At each step it scores every remaining candidate with:
- `relevance_weight * relevance`
- `+ novelty_weight * novelty`
- `+ coverage_weight * coverage`
- `- fairness.penalty_weight * fairness_penalty`
- It then picks the candidate with the highest total value, adds it to the list, updates the running novelty/coverage/fairness state, and repeats until `k_out` items are selected.

#### Exposure fairness
- Let `p` be the **actual exposure distribution** of the current ranked list across groups such as categories.
- Let `q` be the **target distribution** we want to match.
- In the current code:
- `p` is built from the selected top-K list after applying position weights.
- `q` is derived from the reference candidate set:
  - `fairness_kl_pool`: reference is the reranker's top-`pool_size` candidate mix
  - `fairness_kl_full`: reference is the full impression candidate mix
  - If `category_target: uniform`, then `q` is uniform over categories present in the reference set (not used in this project).
  - If `category_target: catalog`, then `q` is the empirical category distribution of that reference set.

#### Worked examples for novelty, coverage, and fairness
- Suppose the reranker has already selected two items: `A` and `B`.
- Candidate `C` has ranker relevance score `0.80`, category `Sports`, entities `{Messi, Inter Miami}`, and is marked as a new item.
- Candidate `D` has relevance score `0.78`, category `Health`, entities `{WHO, vaccine}`, and is not new.

- **Novelty example with `teacher_cosine`**:
  - If `C` has teacher-embedding cosine similarities `0.90` to `A` and `0.35` to `B`, then `novelty(C) = -max(0.90, 0.35) = -0.90`.
  - If `D` has similarities `0.20` to `A` and `0.10` to `B`, then `novelty(D) = -0.20`.
  - Because `-0.20 > -0.90`, `D` is treated as more novel than `C`.

- **Coverage example**:
  - Suppose the selected list has already covered categories `{Sports, Politics}` and entities `{Messi, Real Madrid}`.
  - If `coverage.category_bonus = 1.0`, `coverage.entity_bonus = 0.3`, and `max_new_entities_per_item = 3`:
  - `C` is in `Sports`, which is already covered, so it gets no category bonus. It adds one new entity, `Inter Miami`, so `coverage(C) = 0.3`.
  - `D` is in `Health`, which is new, so it gets `1.0` category bonus. If both `WHO` and `vaccine` are new, it also gets `2 * 0.3 = 0.6` entity bonus, so `coverage(D) = 1.6`.

- **Exposure fairness example**:
  - Suppose the current top-3 list has categories `[Sports, Sports, Health]`.
  - With log position weights, a typical exposure pattern is roughly `[1.00, 0.63, 0.50]`.
  - Then the actual category exposure map is:
  - `Sports: 1.00 + 0.63 = 1.63`
  - `Health: 0.50`
  - After normalization, this becomes `p`, the actual exposure share by category.
  - If the reference candidate pool category mix is `Sports: 50%`, `Health: 30%`, `Politics: 20%`, that normalized mix is `q`.
  - The fairness penalty compares `p` and `q` using both KL divergence and L1 distance.

- **New-item exposure penalty example**:
  - Suppose `new_item_floor = 0.20`.
  - If only the rank-3 item is new, then new-item exposure is `0.50`.
  - Total exposure is `1.00 + 0.63 + 0.50 = 2.13`.
  - So `new_item_exposure_frac = 0.50 / 2.13 = 0.235`, which is above the floor, so no extra penalty is added.
  - If no selected item is new, then `new_item_exposure_frac = 0.0`, which is below `0.20`, so the fairness penalty is increased.

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

## 2.2) What `run_preprocess()` does

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

The current reranker search reports three views of the tradeoff surface on dev:
- `best_feasible`: maximize `nDCG@10` subject to absolute guardrails
- `best_scalar_utility`: maximize a normalized scalar utility
- `pareto_frontier`: nondominated settings across ranking/diversity/fairness axes

The current absolute guardrails are:
- `nDCG@10` must be at least `0.327`.
- `new_item_exposure_frac` must be at least `0.55`.
- `category_coverage@10` must be at least `6.40`.
- `fairness_kl_pool` must be at most `0.241`.

Each candidate in `rerank_search.json` now reports:
- `constraint.feasible`

The scalar utility is computed from absolute-guardrail-normalized units:
- `ndcg_vs_floor_units = nDCG@10 / min_ndcg@k`
- `new_item_exposure_vs_floor_units = new_item_exposure_frac / min_new_item_exposure_frac`
- `category_coverage_vs_floor_units = category_coverage / min_category_coverage`
- `fairness_kl_pool_vs_ceiling_units = (max_fairness_kl_pool - fairness_kl_pool) / max_fairness_kl_pool`

with coefficients:
- `4.0 * ndcg_vs_floor_units`
- `1.5 * new_item_exposure_vs_floor_units`
- `1.0 * category_coverage_vs_floor_units`
- `1.0 * fairness_kl_pool_vs_ceiling_units`

The current selected setting is:
- `relevance_weight=0.90`
- `novelty_weight=0.05`
- `coverage_weight=0.05`
- `novelty_sim=teacher_cosine`
- `fairness.penalty_weight=0.5`
- `fairness.new_item_floor=0.20`

The search writes its summary to `runs/<run_name>/eval/rerank_search.json`.

Artifacts go to `runs/<run_name>/`.

### 3.6 Current demo results (`runs/mind_small_demo`)

Teacher retrieval:
- `recall@200 = 0.02893`

Student ranker:
- `nDCG@10 = 0.33317`
- `MRR = 0.28925`
- `AUC = 0.56954`
- calibration improved `Brier` from `0.1073` to `0.0692`

Feasible reranker operating point:
- `nDCG@10 = 0.32740`
- `new_item_exposure_frac = 0.57631`
- `category_coverage@10 = 6.4409`
- `fairness_kl_pool = 0.24024`

Search summary:
- `best_feasible` matches the current default rerank config
- `best_scalar_utility` is a more aggressive diversity/fairness point, but it is not feasible under the current guardrails

---

## 4) Repo layout

- `src/mindrec/`
  - `data/`: parsing + feature building
  - `models/teacher.py`: teacher-side embedding utilities
  - `models/dlrm.py`: student DLRM ranker + attention fusion
  - `rerank/`: diversity + coverage + exposure fairness reranking
  - `metrics/`: ranking, calibration, diversity, fairness
  - `cli.py`: entrypoint for scripts
- `configs/`: YAML configs
