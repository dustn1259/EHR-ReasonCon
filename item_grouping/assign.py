"""
Assign items to an induced ontology.

Two passes, mirroring how Stage-1 reads the catalog: GROUP first over every
item, then SUBGROUP within each group over only that group's items.

Shares the parsing, normalization and caching conventions of `classify.py`, so
an interrupted run resumes and a human decision is never overwritten.
"""
import hashlib
import json
import os
import re

from classify import _norm, parse_response


def fingerprint(allowed):
    """Short digest of a category list. Used to namespace the cache so two
    different ontologies never share one cache file."""
    joined = "|".join(sorted(str(a) for a in allowed))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


def _slug(name):
    """Filesystem-safe fragment of a category name. Real group names contain
    '/' and '&' ("Respiratory / Ventilation & Gas Monitoring"), which would
    otherwise turn the cache filename into a path into a missing directory."""
    out = [c if (c.isalnum() or c in "-_") else "_" for c in str(name).lower()]
    return re.sub(r"_+", "_", "".join(out)).strip("_")[:40]

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "prompt", "assign_items.txt")
BATCH_SIZE = int(os.getenv("ITEM_GROUPING_BATCH_SIZE", "100"))


def _cache_path(cache_dir, key):
    return os.path.join(cache_dir, f"assign_{key}.json")


def _load(cache_dir, key):
    path = _cache_path(cache_dir, key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save(cache_dir, key, cache):
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_path(cache_dir, key), "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=1)


def build_prompt(level, scope, categories, items):
    template = open(PROMPT_PATH, encoding="utf-8").read()
    body = "\n".join(
        f"{label} : {values}" if str(values).strip() not in ("", "[]", "{}", "nan")
        else str(label)
        for label, values in items
    )
    return (template
            .replace("<<<LEVEL>>>", level)
            .replace("<<<SCOPE>>>", scope)
            .replace("<<<CATEGORIES>>>", categories)
            .replace("<<<INPUT>>>", body))


def item_key(label, linksto):
    """Cache key for one item.

    The source table is part of the key on purpose. 501 labels in MIMIC-III
    occur in more than one source table and 352 of them belong in different
    groups -- "Acetaminophen" is a drug in PRESCRIPTIONS but a serum assay in
    LABEVENTS. Keying on the label alone collapses those into one answer.
    """
    return f"{_norm(label)}\t{str(linksto).strip().lower()}"


def assign(df, mask, column, allowed, level, scope, categories,
           client, model, cache_dir, cache_key,
           batch_size=BATCH_SIZE, log=print):
    """Fill `column` for the rows selected by `mask`.

    `mask` must select rows from a single source table, so the prompt can tell
    the model which table the items come from. Only values on `allowed` are
    written; anything else (Unknown, hallucinated, omitted) is left blank for
    human review. Returns the number filled.
    """
    from models import generate

    cache = _load(cache_dir, cache_key)
    # Never trust a cached label blindly: the same item may have been labelled
    # under a different ontology. Anything not on the current allowed list is
    # ignored and re-asked.
    valid = {v.lower(): v for v in allowed}
    keys = df.loc[mask].apply(lambda r: item_key(r["LABEL"], r["LINKSTO"]), axis=1)

    def cached(key):
        return valid.get(str(cache.get(key, "")).lower(), "")

    resolved = keys.map(cached)
    hit = resolved != ""
    if hit.any():
        df.loc[resolved[hit].index, column] = resolved[hit]
        log(f"    {int(hit.sum())} restored from cache")

    todo_idx = df.index[mask & (df[column].fillna("") == "")]
    if len(todo_idx) == 0:
        return 0

    rows = df.loc[todo_idx]
    items = list(zip(rows["LABEL"], rows["TOP_VALUES"], rows["LINKSTO"]))
    n_batches = (len(items) + batch_size - 1) // batch_size
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompt = build_prompt(level, scope, categories,
                              [(lbl, val) for lbl, val, _ in batch])
        mapping = parse_response(generate(client, prompt, model=model), allowed)
        got = 0
        for label, _, linksto in batch:
            value = mapping.get(_norm(label))
            if value:
                cache[item_key(label, linksto)] = value
                got += 1
        _save(cache_dir, cache_key, cache)
        log(f"    batch {start // batch_size + 1}/{n_batches}: {got}/{len(batch)}")

    resolved = keys.map(cached)
    fill = (resolved != "") & (df.loc[mask, column].fillna("") == "")
    df.loc[resolved[fill].index, column] = resolved[fill]
    return int(fill.sum())


# ------------------------------------------------------------------ passes --

SOURCE_SCOPE = (
    "You are organizing every item recorded for a patient in an ICU database: "
    "vital signs, laboratory results, medications and fluids, intake and output, "
    "procedures, microbiology, and administrative entries.\n"
    "Every item below is recorded in the {table} table, which is what the item "
    "means clinically -- the same name in a different table can be a different "
    "thing (a drug in PRESCRIPTIONS vs. a serum assay in LABEVENTS)."
)


def _sources(df, mask):
    """Iterate (linksto, mask) so each batch comes from one source table."""
    for linksto in df.loc[mask, "LINKSTO"].dropna().unique():
        yield linksto, mask & (df["LINKSTO"] == linksto)


def assign_groups(df, onto, client, model, cache_dir, batch_size=BATCH_SIZE, log=print):
    groups = onto.group_names()
    mask = df["LABEL"].notna()
    total = 0
    for linksto, sub in _sources(df, mask):
        log(f"  {linksto}: {int(sub.sum())} item(s)")
        total += assign(df, sub, "GROUP", groups, "group",
                        SOURCE_SCOPE.format(table=linksto),
                        onto.as_prompt_list(), client, model, cache_dir,
                        f"group_{fingerprint(groups)}", batch_size=batch_size, log=log)
    filled = int((df.loc[mask, "GROUP"].fillna("") != "").sum())
    log(f"  {filled}/{int(mask.sum())} item(s) have a group ({total} new this run)")
    return filled


def assign_subgroups(df, onto, client, model, cache_dir, batch_size=BATCH_SIZE, log=print):
    total = 0
    for group in onto.group_names():
        allowed = onto.subgroup_names(group)
        in_group = df["GROUP"] == group
        if not allowed or not in_group.any():
            continue
        log(f"  {group}: {int(in_group.sum())} item(s)")
        key = f"subgroup_{_slug(group)}_{fingerprint(allowed)}"
        # Batched per group, not per (group, source). The drug-vs-assay
        # ambiguity that makes the source table matter is already resolved by
        # the GROUP pass; splitting again here would only produce a swarm of
        # one-item batches, each costing a full call. Cache keys stay
        # source-aware regardless of how batches are cut.
        scope = (f"Every item below has already been assigned to the category "
                 f"\"{group}\". Choose its subcategory within that group.")
        total += assign(df, in_group, "SUBGROUP", allowed, "subgroup", scope,
                        onto.as_prompt_list(group=group), client, model,
                        cache_dir, key, batch_size=batch_size, log=log)
    filled = int((df["SUBGROUP"].fillna("") != "").sum())
    log(f"  {filled}/{len(df)} item(s) have a subgroup ({total} new this run)")
    return filled


def items_by_group(df):
    """{group: [(label, top_values)]} for subgroup induction."""
    out = {}
    for group, rows in df[df["GROUP"].fillna("") != ""].groupby("GROUP"):
        out[group] = list(zip(rows["LABEL"], rows["TOP_VALUES"]))
    return out
