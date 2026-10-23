"""Turn raw sample metadata into the columns a model needs.

Each output column is declared in the prepare config and derived from one
metadata column. Nothing is assigned by position in the file, and every rule
can be checked against counts you know from the data source (for example the
numbers stated on a GEO record).

Column types
------------
category  text matched against regular expressions, one class per sample:
              label:
                type: category
                from: source_name_ch1
                rules: {normal: "(?i)normal", cancer: "(?i)tumou?r|carcinoma"}
                expected: {normal: 10, cancer: 176}
          Without ``rules`` the raw text values are used as classes.
          ``map: {0: normal, 1: cancer}`` translates coded values instead.
number    first number in the text (``"22.5 mm"`` gives 22.5)
event     survival event: 1/yes/dead/deceased/event/true or 0/no/alive/censored/false;
          anything else is missing. Custom words via ``true_values``/``false_values``.
text      kept as text

``from: _id`` uses the sample ID itself. ``missing_when: {field: F, pattern: R}``
sets the value to missing wherever field F (or ``_id``) matches the regular
expression R, for values known to be unusable, such as numbers imputed upstream.

Values such as "NA", "N/A", "unknown" or an empty string are treated as missing
for every type (override with ``na_values``). Missing categories stay missing;
they are never turned into a class called "NA".
group     patient identifier for keeping related samples together in splits.
          ``strip_regex`` removes a replicate suffix from e.g. the sample title,
          ``expected_groups_with_replicates`` checks the count.
value     a constant for every sample, e.g. ``{value: cancer}``.

Missing outcomes are never filled in. Samples with a missing target are
excluded at training time and counted in the training report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import to_number

NA_STRINGS = {"", "na", "n/a", "nan", "none", "null", "missing", "unknown", "not available", "--"}
TRUE_EVENT = {"1", "1.0", "yes", "dead", "deceased", "event", "true", "died"}
FALSE_EVENT = {"0", "0.0", "no", "alive", "censored", "false", "living"}


class SpecError(ValueError):
    pass


def parse_event(value, true_values=None, false_values=None) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    v = str(value).strip().lower()
    if v in (true_values or TRUE_EVENT):
        return 1.0
    if v in (false_values or FALSE_EVENT):
        return 0.0
    return np.nan


def category_from_rules(text: pd.Series, rules: dict[str, str], expected: dict | None = None) -> pd.Series:
    text = text.fillna("").astype(str)
    hits = pd.DataFrame({name: _contains(text, pat) for name, pat in rules.items()})
    n = hits.sum(axis=1)
    if (n == 0).any():
        bad = text[n == 0]
        raise SpecError(f"{len(bad)} samples match no rule, e.g. {bad.head(5).to_dict()}")
    if (n > 1).any():
        bad = text[n > 1]
        raise SpecError(f"{len(bad)} samples match more than one rule, e.g. {bad.head(5).to_dict()}")
    out = hits.idxmax(axis=1)
    check_expected(out, expected)
    return out


def check_expected(values: pd.Series, expected: dict | None) -> None:
    if not expected:
        return
    got = values.value_counts().to_dict()
    for k, v in expected.items():
        if got.get(k, 0) != v:
            raise SpecError(f"expected {v} samples of '{k}', found {got.get(k, 0)}. All counts: {got}")


def patient_groups(text: pd.Series, strip_regex: str | None) -> tuple[pd.Series, dict]:
    t = text.astype(str).str.strip()
    g = t.str.replace(strip_regex, "", regex=True) if strip_regex else t
    sizes = g.value_counts()
    rep = {"n_samples": int(len(t)), "n_groups": int(len(sizes)),
           "groups_with_more_than_one_sample": int((sizes > 1).sum()),
           "max_group_size": int(sizes.max()) if len(sizes) else 0}
    return g, rep


def derive_columns(metadata: pd.DataFrame, spec: dict | None) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=metadata.index)
    report: dict = {}
    for name, s in (spec or {}).items():
        if not isinstance(s, dict):
            raise SpecError(f"column '{name}': expected a mapping, got {s!r}")
        if "value" in s:
            out[name] = s["value"]
            continue
        src = s.get("from")
        if src == "_id":                     # the sample ID itself (index of the metadata)
            metadata = metadata.assign(_id=metadata.index.astype(str))
        if src not in metadata.columns:
            avail = sorted(metadata.columns)
            raise SpecError(f"column '{name}': metadata field '{src}' not found. Available fields: {avail}")
        raw = metadata[src]
        # Text such as "NA" or "unknown" means missing, not a category (GEO often writes NA).
        na = {str(x).lower() for x in s.get("na_values", NA_STRINGS)}
        raw = raw.where(~raw.astype(str).str.strip().str.lower().isin(na))
        mw = s.get("missing_when")
        if mw:
            # values known to be unusable (e.g. imputed upstream) are set to missing
            fld = mw["field"]
            ref = pd.Series(metadata.index.astype(str), index=metadata.index) if fld == "_id" else metadata[fld].astype(str)
            hit = _contains(ref, mw["pattern"]).to_numpy()
            raw = raw.where(~hit)
            report.setdefault("_missing_when", {})[name] = int(hit.sum())
        kind = s.get("type", "text")
        if kind == "category" and s.get("map"):
            # explicit value -> class mapping, e.g. {0: normal, 1: cancer}; numbers match as numbers
            mapping = {_canon(k): v for k, v in s["map"].items()}
            out[name] = raw.map(lambda v: mapping.get(_canon(v), np.nan) if pd.notna(v) else np.nan)
            unmapped = raw.notna() & out[name].isna()
            if unmapped.any():
                raise SpecError(f"column '{name}': values {sorted(map(str, raw[unmapped].unique()))[:10]} are not in map")
            check_expected(out[name], s.get("expected"))
            report[name] = {"type": kind, "from": src, "n_missing": int(out[name].isna().sum()),
                            "counts": {str(k): int(v) for k, v in out[name].value_counts(dropna=False).items()}}
            continue
        if kind == "category":
            if s.get("rules") and raw.isna().any():
                raise SpecError(f"column '{name}': {int(raw.isna().sum())} samples have a missing value in '{src}', "
                                "so no rule can classify them")
            if s.get("rules"):
                out[name] = category_from_rules(raw, s["rules"], s.get("expected"))
            else:
                out[name] = raw.astype("string").str.strip()
                check_expected(out[name], s.get("expected"))
        elif kind == "number":
            out[name] = raw.map(to_number)
        elif kind == "event":
            tv = {str(x).lower() for x in s["true_values"]} if "true_values" in s else None
            fv = {str(x).lower() for x in s["false_values"]} if "false_values" in s else None
            out[name] = raw.map(lambda x: parse_event(x, tv, fv))
        elif kind == "text":
            out[name] = raw.astype("string")
        elif kind == "group":
            g, grep = patient_groups(raw, s.get("strip_regex"))
            out[name] = g
            exp = s.get("expected_groups_with_replicates")
            if exp is not None and grep["groups_with_more_than_one_sample"] != exp:
                raise SpecError(f"column '{name}': {grep['groups_with_more_than_one_sample']} groups have more than "
                                f"one sample, expected {exp}. Check strip_regex against the values of '{src}'.")
            report[name] = grep
            continue
        else:
            raise SpecError(f"column '{name}': unknown type '{kind}'")
        col = out[name]
        report[name] = {"type": kind, "from": src, "n_missing": int(col.isna().sum())}
        if kind in ("category", "event"):
            report[name]["counts"] = {str(k): int(v) for k, v in col.value_counts(dropna=False).items()}
    return out, report


def _contains(text: pd.Series, pattern: str) -> pd.Series:
    """Regex search; groups in the pattern are allowed (no pandas warning)."""
    import re
    rx = re.compile(pattern)
    return text.astype(str).map(lambda t: rx.search(t) is not None)


def _canon(v):
    """1, 1.0 and "1" compare equal; other values compare as stripped text."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v).strip()


def category_values(col: pd.Series) -> np.ndarray:
    """Categorical covariate values as text, missing kept as NaN. Numbers read
    back from CSV as floats (1.0) are written as integers (1)."""
    def fmt(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    return np.array([fmt(v) for v in col], dtype=object)
