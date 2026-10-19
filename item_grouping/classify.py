"""
Stage B -- assign SUBGROUP to every item in a source catalog.

Rule-based items are labelled first (free, deterministic); whatever is left
goes to the LLM in fixed-size batches. Each batch is cached to
`cache/{source}_llm.json` keyed by item label, so a re-run resumes instead of
re-paying for work already done -- the same convention as the resumable
`concheck_result_*.json` files in Stage 2.

Anything the LLM returns that is not on the source's subgroup list (including
"Unknown", hallucinated names, and items it silently dropped) is left blank and
falls through to `review.py`.
"""
import json
import os
import re

from taxonomy import SOURCES, PROCEDURE_RULES, UNKNOWN, group_for

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "prompt", "classify_items.txt")
BATCH_SIZE = int(os.getenv("ITEM_GROUPING_BATCH_SIZE", "100"))


def _norm(s):
    """Normalize an item name for matching LLM output back to the catalog."""
    s = str(s).strip()
    s = re.sub(r"^[-*•]\s*", "", s)      # strip list bullet
    s = re.sub(r"^\d+[.)]\s*", "", s)          # strip index the model may add
    s = re.sub(r"\s+", " ", s)
    return s.lower()


# ------------------------------------------------------------------ rules ---

def apply_rules(df, source_name):
    """Label items covered by the curated keyword map. Returns rows labelled."""
    if source_name != "procedureevents_mv":
        return 0
    lookup = {}
    for subgroup, labels in PROCEDURE_RULES.items():
        for label in labels:
            lookup[_norm(label)] = subgroup
    hit = df["LABEL"].map(lambda x: lookup.get(_norm(x), ""))
    mask = (df["SUBGROUP"].fillna("") == "") & (hit != "")
    df.loc[mask, "SUBGROUP"] = hit[mask]
    return int(mask.sum())


# -------------------------------------------------------------------- LLM ---

def build_prompt(source, items):
    """items: list of (label, top_values_repr)."""
    template = open(PROMPT_PATH, encoding="utf-8").read()
    subgroup_list = "\n".join(f"{i}. {s}" for i, s in enumerate(source.subgroups, 1))
    body = "\n".join(f"{label} : {values}" for label, values in items)
    return (template
            .replace("<<<TABLE>>>", source.name)
            .replace("<<<VALUE_DESC>>>", source.value_desc)
            .replace("<<<SUBGROUP_LIST>>>", subgroup_list)
            .replace("<<<INPUT>>>", body))


def parse_response(text, valid_subgroups):
    """Parse '- item name : subgroup' lines into {normalized label: subgroup}.

    Subgroups are matched case-insensitively against the allowed list; anything
    else (including Unknown) is dropped so the item lands in the review queue.
    """
    allowed = {s.lower(): s for s in valid_subgroups}
    mapping = {}
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        label, _, subgroup = line.rpartition(":")
        subgroup = subgroup.strip()
        canonical = allowed.get(subgroup.lower())
        if canonical is None:
            continue
        mapping[_norm(label)] = canonical
    return mapping


def _cache_path(cache_dir, source_name):
    return os.path.join(cache_dir, f"{source_name}_llm.json")


def load_cache(cache_dir, source_name):
    path = _cache_path(cache_dir, source_name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_cache(cache_dir, source_name, cache):
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_path(cache_dir, source_name), "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=1)


def classify_with_llm(df, source_name, client, model, cache_dir,
                      batch_size=BATCH_SIZE, limit=None):
    """Fill blank SUBGROUPs using the LLM. Returns (labelled, attempted)."""
    from models import generate

    source = SOURCES[source_name]
    cache = load_cache(cache_dir, source_name)

    # apply anything already cached from a previous run
    cached = df["LABEL"].map(lambda x: cache.get(_norm(x), ""))
    mask = (df["SUBGROUP"].fillna("") == "") & (cached != "")
    df.loc[mask, "SUBGROUP"] = cached[mask]
    print(f"  {int(mask.sum())} item(s) restored from cache")

    todo = df[df["SUBGROUP"].fillna("") == ""]
    if limit:
        todo = todo.iloc[:limit]
    if todo.empty:
        return 0, 0

    items = list(zip(todo["LABEL"], todo["TOP_VALUES"]))
    labelled = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompt = build_prompt(source, batch)
        text = generate(client, prompt, model=model)
        mapping = parse_response(text, source.subgroups)

        got = 0
        for label, _ in batch:
            subgroup = mapping.get(_norm(label))
            if subgroup:
                cache[_norm(label)] = subgroup
                got += 1
        labelled += got
        save_cache(cache_dir, source_name, cache)
        print(f"  batch {start // batch_size + 1}/"
              f"{(len(items) + batch_size - 1) // batch_size}: "
              f"{got}/{len(batch)} labelled")

    resolved = df["LABEL"].map(lambda x: cache.get(_norm(x), ""))
    mask = (df["SUBGROUP"].fillna("") == "") & (resolved != "")
    df.loc[mask, "SUBGROUP"] = resolved[mask]
    return labelled, len(items)


# ------------------------------------------------------------------ groups --

def fill_groups(df, source_name):
    df["GROUP"] = df["SUBGROUP"].map(
        lambda s: group_for(s, source_name) if str(s).strip() else ""
    )
    return df


def pending(df):
    """Rows still needing a human decision."""
    subgroup = df["SUBGROUP"].fillna("").astype(str).str.strip()
    return df[(subgroup == "") | (subgroup == UNKNOWN)]
