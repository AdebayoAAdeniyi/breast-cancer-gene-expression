"""Command line interface.

    exprkit inspect   FILE                   look inside a GEO series matrix, expression table or annotation
    exprkit template  prepare|train          print a commented config template
    exprkit prepare   CONFIG                 raw files -> prepared dataset
    exprkit combine   --data A --data B ...  stack prepared datasets on exact gene symbols
    exprkit batch-check --data D             how well can batches (studies) be told apart?
    exprkit select-genes CONFIG            build a gene list across datasets (occurrence, importance, correlation)
    exprkit train     CONFIG                 evaluate, fit and save a model
    exprkit predict   --model M --data D     apply a saved model to new samples
    exprkit evaluate  --model M --data D     score a saved model on new labelled samples
"""
from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

import pandas as pd
import yaml


def _load_yaml(path):
    p = Path(path)
    with open(p) as fh:
        return yaml.safe_load(fh), p.parent


def _where(args_where):
    out = {}
    for w in args_where or []:
        k, v = w.split("=", 1)
        out.setdefault(k, []).extend(v.split(","))
    return out


def cmd_inspect(a):
    from .io import is_series_matrix, read_annotation, read_matrix, read_series_matrix
    from .transform import scale_summary
    path = Path(a.file)
    fmt = a.format or ("geo_series_matrix" if is_series_matrix(path) else "table")
    if fmt == "geo_series_matrix":
        expr, meta = read_series_matrix(path)
        print(f"{meta.shape[0]} samples; expression table {expr.shape[0]} features x {expr.shape[1]} samples"
              + ("" if not expr.empty else " (empty: values are in a supplementary file)"))
        if not expr.empty:
            print("first feature IDs:", expr.index[:5].tolist())
            print("scale:", scale_summary(expr.T))
        for col in meta.columns:
            vc = meta[col].value_counts(dropna=False)
            if len(vc) <= a.max_values:
                print(f"\n[{col}]  {len(vc)} distinct values")
                print(vc.to_string())
            else:
                print(f"\n[{col}]  {len(vc)} distinct values, e.g. {vc.index[:4].tolist()}")
    elif fmt == "table":
        X, rep = read_matrix(path, a.orientation)
        print(f"{X.shape[0]} samples x {X.shape[1]} features ({a.orientation} as read)")
        print("first samples:", X.index[:5].tolist())
        print("first features:", X.columns[:5].tolist())
        if rep["non_numeric_columns_dropped"]:
            print("non numeric columns dropped:", rep["non_numeric_columns_dropped"][:10])
        print("scale:", scale_summary(X))
    elif fmt == "annotation":
        from .io import open_text
        with open_text(path) as fh:
            head = [ln for ln in fh if not ln.startswith(("#", "!", "^"))][:3]
        print("header and first rows:\n" + "".join(head))
        if a.id_column and a.symbol_column:
            ann = read_annotation(path, a.id_column, a.symbol_column)
            print(f"{len(ann)} rows; {(ann.symbol != '').sum()} with a symbol; "
                  f"{ann.symbol.str.contains('///').sum()} mapped to several genes")


def cmd_template(a):
    text = resources.files("exprkit.templates").joinpath(f"{a.kind}.yaml").read_text()
    sys.stdout.write(text)


def cmd_prepare(a):
    from .prepare import prepare
    cfg, base = _load_yaml(a.config)
    prepare(cfg, ".")


def cmd_train(a):
    from .train import train
    cfg, base = _load_yaml(a.config)
    train(cfg, ".")


def cmd_select_genes(a):
    from .selection import run_select_genes
    cfg, _ = _load_yaml(a.config)
    run_select_genes(cfg, ".")


def cmd_combine(a):
    from .integrate import combine_expression
    from .io import load_dataset, save_dataset
    exprs, samples, reports = {}, {}, {}
    for d in a.data:
        e, s, r = load_dataset(d)
        name = Path(d).name
        exprs[name], samples[name], reports[name] = e, s, (e, s, r)
    for name, (_, _, r) in reports.items():
        if r.get("normalise", "none") != "none":
            raise SystemExit(f"{name} is already normalised ({r['normalise']}); combine datasets prepared without "
                             "normalisation, then use --normalise here")
    X, dataset, rep = combine_expression(exprs, a.how)
    from .transform import apply_normalisation
    X = apply_normalisation(X, a.normalise)
    rep["normalise"] = a.normalise
    S = pd.concat(samples.values()).loc[X.index]
    if "dataset" not in S.columns:
        S["dataset"] = dataset.loc[X.index]          # folder name when the inputs carry no dataset column
    save_dataset(a.out, X, S, {"combined_from": a.data, **rep})
    print(f"{X.shape[0]} samples x {X.shape[1]} genes -> {a.out}")


def cmd_batch_check(a):
    from .evaluation import batch_separability
    from .io import load_dataset, write_json
    from .train import select_rows
    X, S, _ = load_dataset(a.data)
    m = select_rows(S, _where(a.where))
    res = batch_separability(X.loc[m], S.loc[m, a.batch_column], seed=a.seed)
    print(res)
    if a.out:
        write_json(res, a.out)


def _check_not_normalised(report, path):
    if report.get("normalise", "none") != "none":
        raise SystemExit(f"{path} was normalised during prepare ({report['normalise']}). Prepare new data without "
                         "normalise: the saved model applies its own normalisation over its training genes.")


def cmd_predict(a):
    from .bundle import load_bundle, predict, prepare_input
    from .io import load_dataset, read_matrix, write_json
    from .train import select_rows
    from .transform import apply_transform
    b = load_bundle(a.model)
    if a.data:
        X, S, r = load_dataset(a.data)
        _check_not_normalised(r, a.data)
        m = select_rows(S, _where(a.where))
        X, S = X.loc[m], S.loc[m]
    else:
        X, _ = read_matrix(a.matrix, a.orientation)
        X = apply_transform(X, a.transform)
        S = pd.read_csv(a.samples, index_col=0).reindex(X.index) if a.samples else None
    Xin, rep = prepare_input(b, X, S, a.max_missing, a.allow_missing_covariates)
    pred = predict(b, Xin)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out)
    write_json(rep, str(Path(a.out).with_suffix("")) + "_input_report.json")
    print(f"{len(pred)} predictions -> {a.out}")


def cmd_evaluate(a):
    from .bundle import evaluate_external, load_bundle, prepare_input
    from .io import load_dataset, write_json
    from .train import select_rows
    b = load_bundle(a.model)
    X, S, r = load_dataset(a.data)
    _check_not_normalised(r, a.data)
    m = select_rows(S, _where(a.where))
    X, S = X.loc[m], S.loc[m]
    Xin, rep = prepare_input(b, X, S, a.max_missing, a.allow_missing_covariates)
    res = evaluate_external(b, Xin, S, a.out, a.bootstrap, a.seed)
    res["input_report"] = rep
    write_json(res, Path(a.out) / "external_evaluation.json")
    print({k: v for k, v in res.items() if k.startswith(("bootstrap", "n_used"))})


def main(argv=None):
    p = argparse.ArgumentParser(prog="exprkit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect", help="look inside an input file")
    s.add_argument("file")
    s.add_argument("--format", choices=["geo_series_matrix", "table", "annotation"])
    s.add_argument("--orientation", default="genes_by_samples", choices=["genes_by_samples", "samples_by_genes"])
    s.add_argument("--id-column")
    s.add_argument("--symbol-column")
    s.add_argument("--max-values", type=int, default=12)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("template", help="print a config template")
    s.add_argument("kind", choices=["prepare", "train", "select-genes"])
    s.set_defaults(func=cmd_template)

    for name, fn in (("prepare", cmd_prepare), ("train", cmd_train), ("select-genes", cmd_select_genes)):
        s = sub.add_parser(name, help=f"{name} from a YAML config")
        s.add_argument("config", help="YAML file; relative paths inside it are read from the current folder")
        s.set_defaults(func=fn)

    s = sub.add_parser("combine", help="stack prepared datasets on exact gene symbols")
    s.add_argument("--data", action="append", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--how", default="inner", choices=["inner", "outer"])
    s.add_argument("--normalise", default="none", choices=["none", "per_sample_rank"],
                   help="per_sample_rank makes different platforms comparable (applied after combining)")
    s.set_defaults(func=cmd_combine)

    s = sub.add_parser("batch-check", help="cross validated AUC for telling two batches apart")
    s.add_argument("--data", required=True)
    s.add_argument("--batch-column", default="dataset")
    s.add_argument("--where", action="append", help="column=value[,value] to restrict rows, e.g. label=cancer")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_batch_check)

    for name, fn in (("predict", cmd_predict), ("evaluate", cmd_evaluate)):
        s = sub.add_parser(name, help=f"{name} with a saved model")
        s.add_argument("--model", required=True)
        s.add_argument("--data", help="prepared dataset folder", required=(name == "evaluate"))
        s.add_argument("--where", action="append")
        s.add_argument("--out", required=True, help="predictions CSV (predict) or output folder (evaluate)")
        s.add_argument("--max-missing", type=float, default=0.2,
                       help="largest allowed fraction of model genes missing from the new data")
        s.add_argument("--allow-missing-covariates", action="store_true")
        if name == "predict":
            s.add_argument("--matrix", help="expression table instead of a prepared dataset")
            s.add_argument("--orientation", default="genes_by_samples", choices=["genes_by_samples", "samples_by_genes"])
            s.add_argument("--transform", default="none", choices=["none", "log2", "log2p1", "log2cpm"])
            s.add_argument("--samples", help="CSV with covariates, first column = sample ID")
        else:
            s.add_argument("--bootstrap", type=int, default=1000)
            s.add_argument("--seed", type=int, default=0)
        s.set_defaults(func=fn)

    a = p.parse_args(argv)
    if a.cmd == "predict" and not (a.data or a.matrix):
        p.error("predict needs --data or --matrix")
    a.func(a)


if __name__ == "__main__":
    main()
