# Metrics in this repo

## Ranking quality (per impression)
- **AUC**: discrimination between clicked vs non-clicked candidates within impressions.
- **MRR**: reciprocal rank of the first clicked item.
- **nDCG@K**: position-aware gain for clicked items; normalized by ideal DCG.
- **Recall@K**: fraction of clicked items captured in top-K.
- **MAP@K**: precision integrated over ranks up to K.

## Calibration (global, over all scored pairs)
- **Brier score**: mean squared error of predicted probability vs label.
- **ECE (Expected Calibration Error)**: bins predictions; compares bin accuracy vs confidence.

## Diversity (list-level, top-K)
- **ILD** (intra-list diversity): 1 - average pairwise similarity (teacher cosine) within the list.
- **Category coverage@K**: number of unique categories in top-K.
- **Category entropy@K**: entropy of category distribution in top-K (higher = more spread).

## Exposure fairness (list-level, top-K)
Position-weighted exposure uses a bias curve v(pos) (log or linear).
- **KL / L1 disparity**: compare exposure distribution vs target distribution (catalog or uniform).
- **Gini**: inequality of exposure allocation across groups.
- **New-item exposure fraction**: how much position-weighted exposure is allocated to items tagged as new/rare.

Target definition note:
- `catalog` target means the empirical category mix of the impression candidate pool.
- `uniform` target means equal mass across the categories present in that impression candidate pool.
- The target is not derived from the selected top-K list, and `catalog` here is not a global full-corpus category prior.
