"""
Stage C -- human review.

This replaces the manual pass that used to happen inside the notebook (eyeball
`df[df['SUBGROUP'] == '']`, retype values, re-run a slice). Two workflows:

  export/apply  write the rows needing a decision to a review CSV, edit the
                SUBGROUP column in any spreadsheet, then merge it back.
                Suited to bulk review by a clinician.

  interactive   walk the same rows in the terminal, picking a subgroup by
                number. Suited to a short queue.

Both are idempotent: reviewed decisions are recorded in
`cache/{source}_human.json` and re-applied automatically on later runs, so a
re-classification never silently overwrites a human decision.
"""
import json
import os

import pandas as pd

from classify import _norm, pending
from taxonomy import SOURCES, UNKNOWN

REVIEW_COLS = ["LABEL", "TOP_VALUES", "SUBGROUP", "GROUP", "LINKSTO"]


def _path(cache_dir, source_name):
    return os.path.join(cache_dir, f"{source_name}_human.json")


def load_decisions(cache_dir, source_name):
    path = _path(cache_dir, source_name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_decisions(cache_dir, source_name, decisions):
    os.makedirs(cache_dir, exist_ok=True)
    with open(_path(cache_dir, source_name), "w", encoding="utf-8") as fh:
        json.dump(decisions, fh, ensure_ascii=False, indent=1)


def apply_decisions(df, cache_dir, source_name):
    """Human decisions win over anything the LLM or the rules produced."""
    decisions = load_decisions(cache_dir, source_name)
    if not decisions:
        return 0
    resolved = df["LABEL"].map(lambda x: decisions.get(_norm(x), ""))
    mask = resolved != ""
    df.loc[mask, "SUBGROUP"] = resolved[mask]
    return int(mask.sum())


# ------------------------------------------------------------ export/apply --

def export(df, source_name, out_path, include_all=False):
    """Write the review queue. `include_all` exports every row for a full audit."""
    rows = df if include_all else pending(df)
    rows = rows[REVIEW_COLS].copy()
    rows.insert(0, "ROW", rows.index)
    rows.to_csv(out_path, index=False)
    allowed = SOURCES[source_name].subgroups
    print(f"  wrote {len(rows)} row(s) -> {out_path}")
    if allowed:
        print("  allowed SUBGROUP values:")
        for s in allowed:
            print(f"    - {s}")
    return len(rows)


def apply_file(df, source_name, review_path, cache_dir):
    """Merge an edited review CSV back in, recording each decision."""
    edited = pd.read_csv(review_path)
    if "SUBGROUP" not in edited.columns or "LABEL" not in edited.columns:
        raise SystemExit(f"{review_path} must keep its LABEL and SUBGROUP columns")

    allowed = {s.lower(): s for s in SOURCES[source_name].subgroups}
    decisions = load_decisions(cache_dir, source_name)
    applied, rejected = 0, []
    for label, subgroup in zip(edited["LABEL"], edited["SUBGROUP"]):
        subgroup = str(subgroup).strip()
        if not subgroup or subgroup.lower() in ("nan", UNKNOWN.lower()):
            continue
        # Lab/micro subgroups are metadata-derived and have no fixed list.
        canonical = allowed.get(subgroup.lower(), subgroup if not allowed else None)
        if canonical is None:
            rejected.append((label, subgroup))
            continue
        decisions[_norm(label)] = canonical
        applied += 1

    save_decisions(cache_dir, source_name, decisions)
    if rejected:
        print(f"  [WARN] {len(rejected)} row(s) skipped -- subgroup not on the "
              f"allowed list, e.g. {rejected[:3]}")
    print(f"  recorded {applied} human decision(s)")
    return apply_decisions(df, cache_dir, source_name)


# ------------------------------------------------------------- interactive --

def interactive(df, source_name, cache_dir):
    """Terminal review loop over the pending rows."""
    allowed = SOURCES[source_name].subgroups
    if not allowed:
        raise SystemExit(f"{source_name} has no fixed subgroup list; use export/apply")

    queue = pending(df)
    if queue.empty:
        print("  nothing to review")
        return 0

    decisions = load_decisions(cache_dir, source_name)
    print(f"  {len(queue)} item(s) to review. "
          f"Enter a number, 's' to skip, 'q' to stop and save.\n")
    for i, s in enumerate(allowed, 1):
        print(f"    {i:>2}. {s}")

    reviewed = 0
    for n, (_, row) in enumerate(queue.iterrows(), 1):
        print(f"\n[{n}/{len(queue)}] {row['LABEL']}")
        print(f"        values: {str(row['TOP_VALUES'])[:120]}")
        try:
            answer = input("        subgroup #> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  interrupted -- saving what was reviewed")
            break
        if answer.lower() == "q":
            break
        if answer.lower() in ("", "s"):
            continue
        if not answer.isdigit() or not 1 <= int(answer) <= len(allowed):
            print("        ignored (not a valid number)")
            continue
        decisions[_norm(row["LABEL"])] = allowed[int(answer) - 1]
        reviewed += 1

    save_decisions(cache_dir, source_name, decisions)
    print(f"\n  recorded {reviewed} human decision(s)")
    return apply_decisions(df, cache_dir, source_name)
