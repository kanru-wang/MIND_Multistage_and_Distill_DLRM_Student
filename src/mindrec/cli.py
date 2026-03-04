from __future__ import annotations

import argparse
from pathlib import Path

from mindrec.config import load_config
from mindrec.pipeline.evaluate import run_evaluate
from mindrec.pipeline.preprocess import run_preprocess
from mindrec.pipeline.ranker_train import run_train_ranker
from mindrec.pipeline.rerank_eval import run_rerank_eval
from mindrec.pipeline.retrieval import run_build_index, run_eval_retrieval
from mindrec.pipeline.teacher_train import run_train_teacher


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, type=str, help="Path to YAML config")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mindrec")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess", help="Parse raw MIND TSV into processed parquet")
    _add_config_arg(p)

    p = sub.add_parser(
        "train_teacher", help="Train/compute teacher embeddings (item+user)"
    )
    _add_config_arg(p)

    p = sub.add_parser("build_index", help="Build Faiss ANN index for retrieval")
    _add_config_arg(p)

    p = sub.add_parser(
        "eval_retrieval", help="Evaluate retrieval recall@K on dev impressions"
    )
    _add_config_arg(p)

    p = sub.add_parser(
        "train_ranker", help="Train DLRM student ranker with distillation"
    )
    _add_config_arg(p)

    p = sub.add_parser(
        "evaluate", help="Evaluate ranker on dev impressions (many metrics)"
    )
    _add_config_arg(p)

    p = sub.add_parser(
        "rerank_eval", help="Evaluate diversity+coverage+fairness reranker"
    )
    _add_config_arg(p)

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "preprocess":
        run_preprocess(cfg)
        return
    if args.cmd == "train_teacher":
        run_train_teacher(cfg)
        return
    if args.cmd == "build_index":
        run_build_index(cfg)
        return
    if args.cmd == "eval_retrieval":
        run_eval_retrieval(cfg)
        return
    if args.cmd == "train_ranker":
        run_train_ranker(cfg)
        return
    if args.cmd == "evaluate":
        run_evaluate(cfg)
        return
    if args.cmd == "rerank_eval":
        run_rerank_eval(cfg)
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
