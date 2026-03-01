param(
  [string]$Config="configs/mind_small.yaml"
)

python -m mindrec.cli preprocess --config $Config
python -m mindrec.cli train_teacher --config $Config
python -m mindrec.cli build_index --config $Config
python -m mindrec.cli eval_retrieval --config $Config
python -m mindrec.cli train_ranker --config $Config
python -m mindrec.cli evaluate --config $Config
python -m mindrec.cli rerank_eval --config $Config
