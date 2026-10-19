"""
Group / subgroup classification of MIMIC-III items.

Builds the item catalog that Stage-1's `cluster` NER method browses: every item
recorded for a patient, tagged with a coarse GROUP and a fine SUBGROUP. Output
schema matches `EHR_ReasonCon/patient_data/{hadm}/catalog_items.csv`:

    LABEL  TOP_VALUES  SUBGROUP  GROUP  LINKSTO  ITEMID

Pipeline, per source table:

    build      enumerate items + collect TOP_VALUES from the event table
    classify   rules -> LLM (batched, cached) -> whatever is left is blank
    review     human decides the blanks (spreadsheet or terminal)
    merge      concatenate every source into one catalog

Usage
-----
    export MIMIC_DIR=/path/to/mimiciii

    python build_item_catalog.py build     --source chartevents
    python build_item_catalog.py classify  --source chartevents
    python build_item_catalog.py review    --source chartevents --export
    #   ... edit output/review/chartevents_review.csv ...
    python build_item_catalog.py review    --source chartevents --apply
    python build_item_catalog.py merge

    python build_item_catalog.py all --source chartevents   # build + classify
    python build_item_catalog.py status                     # coverage per source

`--source all` runs every source table in turn.

Environment
-----------
    MIMIC_DIR                      MIMIC-III csv directory
    GOOGLE_APPLICATION_CREDENTIALS / PROJECT_ID / LOCATION   Vertex AI
    GOOGLE_API_KEY                 fallback (Gemini Developer API)
    ITEM_GROUPING_MODEL            default gemini-2.5-pro
    ITEM_GROUPING_BATCH_SIZE       items per LLM call, default 100
"""
import argparse
import os

import pandas as pd

import classify as classify_mod
import review as review_mod
from catalog import build as build_catalog
from taxonomy import SOURCES

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
REVIEW_DIR = os.path.join(OUT_DIR, "review")
MERGED = os.path.join(OUT_DIR, "item_catalog.csv")
ONTOLOGY = os.path.join(OUT_DIR, "ontology.json")
INDUCED = os.path.join(OUT_DIR, "item_catalog_induced.csv")

OUT_COLS = ["LABEL", "TOP_VALUES", "SUBGROUP", "GROUP", "LINKSTO", "ITEMID"]


def catalog_path(source_name):
    return os.path.join(OUT_DIR, f"{source_name}_items.csv")


def load(source_name):
    path = catalog_path(source_name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run `build --source {source_name}` first")
    df = pd.read_csv(path, low_memory=False, keep_default_na=False)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def store(df, source_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(catalog_path(source_name), index=False)


def resolve_sources(name):
    if name == "all":
        return list(SOURCES)
    if name not in SOURCES:
        raise SystemExit(f"unknown source '{name}'. Known: {', '.join(SOURCES)}")
    return [name]


def coverage(df):
    subgroup = df["SUBGROUP"].fillna("").astype(str).str.strip()
    done = int((subgroup != "").sum())
    return done, len(df)


# ------------------------------------------------------------------ stages --

def cmd_build(args):
    for name in resolve_sources(args.source):
        print(f"\n=== build {name}")
        df = build_catalog(name, top_n=args.top_n)
        df = classify_mod.fill_groups(df, name)
        store(df[OUT_COLS], name)
        done, total = coverage(df)
        print(f"  {total} items -> {catalog_path(name)}  ({done} pre-labelled)")


def cmd_classify(args):
    client, model = None, None
    for name in resolve_sources(args.source):
        source = SOURCES[name]
        print(f"\n=== classify {name}  (strategy: {source.strategy})")
        df = load(name)

        n = review_mod.apply_decisions(df, CACHE_DIR, name)
        if n:
            print(f"  {n} item(s) restored from earlier human review")

        n = classify_mod.apply_rules(df, name)
        if n:
            print(f"  {n} item(s) labelled by rules")

        # Checkpoint the free, deterministic labels before spending any tokens,
        # so an LLM/credential failure below never discards them.
        store(classify_mod.fill_groups(df, name)[OUT_COLS], name)

        if source.strategy == "metadata":
            print("  metadata-derived subgroups; no LLM call needed")
        elif args.skip_llm:
            print("  --skip-llm: leaving the rest for review")
        else:
            try:
                if client is None:
                    from models import init_client, DEFAULT_MODEL
                    client, model = init_client(), args.model or DEFAULT_MODEL
                    print(f"  model: {model}")
                classify_mod.classify_with_llm(
                    df, name, client, model, CACHE_DIR,
                    batch_size=args.batch_size, limit=args.limit,
                )
            except Exception as exc:
                # Batch results are cached as they land, so a later run resumes.
                print(f"  [ERROR] LLM classification stopped: "
                      f"{type(exc).__name__}: {exc}")
                if args.source != "all":
                    raise

        df = classify_mod.fill_groups(df, name)
        store(df[OUT_COLS], name)
        done, total = coverage(df)
        print(f"  {done}/{total} labelled, {total - done} left for review")


def cmd_review(args):
    os.makedirs(REVIEW_DIR, exist_ok=True)
    for name in resolve_sources(args.source):
        print(f"\n=== review {name}")
        df = load(name)
        review_mod.apply_decisions(df, CACHE_DIR, name)
        path = args.file or os.path.join(REVIEW_DIR, f"{name}_review.csv")

        if args.apply:
            review_mod.apply_file(df, name, path, CACHE_DIR)
        elif args.interactive:
            review_mod.interactive(df, name, CACHE_DIR)
        else:
            review_mod.export(df, name, path, include_all=args.all_rows)
            continue

        df = classify_mod.fill_groups(df, name)
        store(df[OUT_COLS], name)
        done, total = coverage(df)
        print(f"  {done}/{total} labelled, {total - done} left for review")


def cmd_merge(args):
    frames = []
    for name in SOURCES:
        path = catalog_path(name)
        if not os.path.exists(path):
            print(f"  [skip] {name}: not built")
            continue
        df = load(name)
        review_mod.apply_decisions(df, CACHE_DIR, name)
        df = classify_mod.fill_groups(df, name)
        done, total = coverage(df)
        if done < total:
            print(f"  [WARN] {name}: {total - done} unlabelled item(s) included")
        frames.append(df[OUT_COLS])

    if not frames:
        raise SystemExit("nothing to merge")
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(MERGED, index=False)
    print(f"\n{len(merged)} items -> {MERGED}")
    print(merged.groupby("GROUP").size().sort_values(ascending=False).to_string())


def cmd_status(args):
    print(f"{'source':<22}{'items':>8}{'labelled':>10}{'pending':>9}")
    for name in SOURCES:
        path = catalog_path(name)
        if not os.path.exists(path):
            print(f"{name:<22}{'-':>8}{'-':>10}{'-':>9}   (not built)")
            continue
        df = load(name)
        done, total = coverage(df)
        print(f"{name:<22}{total:>8}{done:>10}{total - done:>9}")


def cmd_all(args):
    cmd_build(args)
    cmd_classify(args)


# --------------------------------------------------- ontology induction -----

def _all_items():
    """Every built item across every source, in a stable order."""
    frames = []
    for name in SOURCES:
        if os.path.exists(catalog_path(name)):
            frames.append(load(name))
    if not frames:
        raise SystemExit("nothing built yet -- run `build --source all` first")
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("LABEL", kind="stable").reset_index(drop=True)


def _llm(args):
    from models import init_client, DEFAULT_MODEL
    return init_client(), (getattr(args, "model", None) or DEFAULT_MODEL)


def cmd_induce(args):
    """Discover the GROUP/SUBGROUP tree from the items themselves."""
    from ontology import Ontology
    import induce as induce_mod
    import assign as assign_mod

    df = _all_items()
    onto = Ontology.load(ONTOLOGY) if not args.restart else Ontology()
    client, model = _llm(args)
    print(f"model: {model}   items: {len(df)}")

    # unset flags fall back to the module defaults in induce.py
    loop = {k: v for k, v in (("seed_batch", args.seed_batch),
                              ("extend_batch", args.extend_batch),
                              ("patience", args.patience)) if v is not None}

    stage = args.stage
    if stage in ("groups", "both"):
        print("\n=== inducing GROUPs")
        items = list(zip(df["LABEL"], df["TOP_VALUES"]))
        induce_mod.induce_groups(items, client, model, onto=onto, **loop)
        onto.save(ONTOLOGY)
        print(f"  -> {onto.summary()}  ({ONTOLOGY})")

    if stage in ("subgroups", "both"):
        if not onto.group_names():
            raise SystemExit("no groups yet -- run `induce --stage groups` first")
        print("\n=== assigning items to GROUPs (needed before subgroup induction)")
        for col in ("GROUP", "SUBGROUP"):
            if col not in df.columns:
                df[col] = ""
        df["GROUP"] = ""
        assign_mod.assign_groups(df, onto, client, model, CACHE_DIR,
                                 batch_size=args.batch_size)
        print("\n=== inducing SUBGROUPs within each group")
        induce_mod.induce_subgroups(assign_mod.items_by_group(df), client, model,
                                    onto, **loop)
        onto.save(ONTOLOGY)
        print(f"  -> {onto.summary()}  ({ONTOLOGY})")

    cmd_ontology(args)


def cmd_assign(args):
    """Assign every item to the induced ontology and write the catalog."""
    from ontology import Ontology
    import assign as assign_mod

    df = _all_items()
    for col in ("GROUP", "SUBGROUP"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if args.use_taxonomy:
        # `build` already fills SUBGROUP/GROUP for the metadata-derived sources
        # (labevents, microbiologyevents). Harvest those into the ontology and
        # keep them on the rows -- re-asking the LLM would only lose them.
        onto = Ontology.from_taxonomy(catalog=df)
        kept = int((df["SUBGROUP"] != "").sum())
        if kept:
            print(f"  {kept} item(s) already labelled by MIMIC metadata; kept as-is")
    else:
        onto = Ontology.load(ONTOLOGY)
        df["GROUP"] = ""
        df["SUBGROUP"] = ""

    if not onto.group_names():
        raise SystemExit(f"{ONTOLOGY} has no groups -- run `induce` first")
    client, model = _llm(args)
    print(f"model: {model}   items: {len(df)}   ontology: {onto.summary()}")

    print("\n=== GROUP pass")
    assign_mod.assign_groups(df, onto, client, model, CACHE_DIR, batch_size=args.batch_size)
    print("\n=== SUBGROUP pass")
    assign_mod.assign_subgroups(df, onto, client, model, CACHE_DIR, batch_size=args.batch_size)

    df[OUT_COLS].to_csv(INDUCED, index=False)
    blank = int((df["SUBGROUP"].fillna("") == "").sum())
    print(f"\n{len(df)} items -> {INDUCED}")
    if blank:
        print(f"  [WARN] {blank} item(s) have no subgroup -- review them")
    print(df.groupby("GROUP").size().sort_values(ascending=False).to_string())


def cmd_patient_catalog(args):
    """Split the global catalog into per-patient patient_records / catalog_items."""
    import patient_split

    path = args.catalog or (INDUCED if os.path.exists(INDUCED) else MERGED)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run `merge` or `assign` first")
    catalog = pd.read_csv(path, low_memory=False, keep_default_na=False)
    catalog = catalog.loc[:, ~catalog.columns.str.startswith("Unnamed")]
    blank = int((catalog["SUBGROUP"].fillna("") == "").sum())
    print(f"catalog: {path}  ({len(catalog)} items"
          + (f", {blank} without a subgroup)" if blank else ")"))

    patient_split.split_all(catalog, args.patient_data, dest_root=args.dest)


def cmd_ontology(args):
    """Print the induced ontology."""
    from ontology import Ontology
    onto = Ontology.load(ONTOLOGY)
    if not onto.group_names():
        raise SystemExit(f"{ONTOLOGY} not found or empty -- run `induce` first")
    print(f"\n{onto.summary()}")
    for group in onto.group_names():
        spec = onto.groups[group]
        print(f"\n{group}")
        if spec.get("description"):
            print(f"    {spec['description']}")
        for sub, desc in spec["subgroups"].items():
            print(f"      - {sub}" + (f"  ({desc})" if desc else ""))


def main():
    parser = argparse.ArgumentParser(
        description="Classify MIMIC-III items into GROUP / SUBGROUP.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source(p):
        p.add_argument("--source", default="all",
                       help=f"one of: {', '.join(SOURCES)}, or 'all'")

    p = sub.add_parser("build", help="enumerate items + collect TOP_VALUES")
    add_source(p)
    p.add_argument("--top-n", type=int, default=10)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("classify", help="rules + LLM classification")
    add_source(p)
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=classify_mod.BATCH_SIZE)
    p.add_argument("--limit", type=int, default=None,
                   help="classify at most N items (smoke test)")
    p.add_argument("--skip-llm", action="store_true",
                   help="rules + metadata only; no LLM calls, no credentials needed")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("review", help="human review of unlabelled items")
    add_source(p)
    p.add_argument("--export", action="store_true", help="write the review CSV (default)")
    p.add_argument("--apply", action="store_true", help="merge an edited review CSV back")
    p.add_argument("--interactive", action="store_true", help="review in the terminal")
    p.add_argument("--all-rows", action="store_true",
                   help="export every row, not just the pending ones")
    p.add_argument("--file", default=None, help="review CSV path")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("merge", help="concatenate all sources into item_catalog.csv")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("status", help="labelled / pending counts per source")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("induce", help="discover the GROUP/SUBGROUP tree from the items")
    p.add_argument("--stage", choices=["groups", "subgroups", "both"], default="both")
    p.add_argument("--restart", action="store_true",
                   help="discard any existing ontology.json and start over")
    p.add_argument("--seed-batch", type=int, default=None)
    p.add_argument("--extend-batch", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="stop after N consecutive batches propose nothing new")
    p.add_argument("--batch-size", type=int, default=classify_mod.BATCH_SIZE)
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_induce)

    p = sub.add_parser("assign", help="assign every item to the induced ontology")
    p.add_argument("--use-taxonomy", action="store_true",
                   help="assign against the fixed taxonomy.py instead of ontology.json")
    p.add_argument("--batch-size", type=int, default=classify_mod.BATCH_SIZE)
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_assign)

    p = sub.add_parser("patient-catalog",
                       help="split the catalog into per-patient files for Stage 1")
    p.add_argument("--patient-data", required=True,
                   help="directory of {hadm}/ dirs holding the per-patient event tables")
    p.add_argument("--dest", default=None,
                   help="write into this root instead of alongside the event tables")
    p.add_argument("--catalog", default=None,
                   help="catalog csv (default: item_catalog_induced.csv, else item_catalog.csv)")
    p.set_defaults(func=cmd_patient_catalog)

    p = sub.add_parser("ontology", help="print the induced ontology")
    p.set_defaults(func=cmd_ontology)

    p = sub.add_parser("all", help="build + classify")
    add_source(p)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=classify_mod.BATCH_SIZE)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-llm", action="store_true")
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
