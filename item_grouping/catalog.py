"""
Stage A -- build the raw item catalog for one MIMIC-III source table.

For every item in the source's dictionary this collects:

    ITEMID  LABEL  TOP_VALUES  LINKSTO  SUBGROUP  GROUP

TOP_VALUES is the item's most frequent recorded values (default: top 10, by
descending frequency). It is shown to the LLM as weak context and to the human
reviewer as a sanity check -- the prompt explicitly tells the model not to
classify on it.

Event tables are counted in chunks: CHARTEVENTS.csv alone is ~33 GB, so it
never gets loaded whole.
"""
import ast
import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

from taxonomy import SOURCES, PRESCRIPTION_ITEMID_BASE, GROUP_LAB

CHUNK_ROWS = 2_000_000
TOP_N = 10


def mimic_path(name):
    default = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mimiciii"))
    return os.path.join(os.getenv("MIMIC_DIR", default), name)


# ------------------------------------------------------------ dictionaries --

def load_dictionary(source):
    """Enumerate the items of one source table as a DataFrame with ITEMID + LABEL."""
    if source.dictionary == "D_ITEMS":
        # D_ITEMS.LINKSTO names the event table the item belongs to. For
        # microbiologyevents the dictionary key is ITEMID but the event table
        # joins on SPEC_ITEMID -- handled by the caller.
        d = pd.read_csv(mimic_path("D_ITEMS.csv"), low_memory=False)
        return d[d["LINKSTO"] == source.name].reset_index(drop=True)

    if source.dictionary == "D_LABITEMS":
        return pd.read_csv(mimic_path("D_LABITEMS.csv"), low_memory=False)

    if source.dictionary == "PRESCRIPTIONS":
        # Prescriptions have no dictionary table -- the distinct DRUG strings
        # are the items. Synthetic ITEMIDs keep the merged catalog unique.
        pres = pd.read_csv(mimic_path("PRESCRIPTIONS.csv"), low_memory=False,
                           usecols=["DRUG"])
        drugs = (pres["DRUG"].dropna().astype(str).str.strip()
                 .replace("", pd.NA).dropna().drop_duplicates()
                 .sort_values().reset_index(drop=True))
        return pd.DataFrame({
            "ITEMID": PRESCRIPTION_ITEMID_BASE + drugs.index + 1,
            "DRUG": drugs,
        })

    raise ValueError(f"unknown dictionary {source.dictionary}")


# -------------------------------------------------------------- top values --

def compute_top_values(source, keys, top_n=TOP_N):
    """{key: [most frequent values]} for the source's event table.

    `keys` is the set of join values (ITEMID / SPEC_ITEMID / DRUG) to collect.
    Multi-column sources (inputevents AMOUNT+RATE) return a dict per key.
    """
    path = mimic_path(source.event_table)
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found -- TOP_VALUES left empty")
        return {}

    keys = set(keys)
    counters = {col: {} for col in source.value_cols}
    usecols = [source.join_col] + source.value_cols

    # Pin the value dtype instead of letting pandas infer it per chunk: an
    # all-numeric first chunk of CHARTEVENTS.VALUE would come back as int64 and
    # a later chunk with free text as object, mixing types within one item.
    dtypes = {col: str for col in source.value_cols} if source.value_dtype == "str" else None

    reader = pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS,
                         dtype=dtypes, low_memory=False)
    for chunk in tqdm(reader, desc=f"  scanning {source.event_table}", unit="chunk"):
        chunk = chunk[chunk[source.join_col].isin(keys)]
        if chunk.empty:
            continue
        for col in source.value_cols:
            sub = chunk[[source.join_col, col]].dropna()
            if sub.empty:
                continue
            if source.value_dtype == "str":
                sub[col] = sub[col].str.strip()
            else:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
                sub = sub.dropna(subset=[col])
            for key, values in sub.groupby(source.join_col)[col]:
                counters[col].setdefault(key, Counter()).update(values.tolist())

    out = {}
    for key in keys:
        if len(source.value_cols) == 1:
            counter = counters[source.value_cols[0]].get(key)
            out[key] = [v for v, _ in counter.most_common(top_n)] if counter else []
        else:
            out[key] = {
                col: [v for v, _ in counters[col].get(key, Counter()).most_common(top_n)]
                for col in source.value_cols
            }
    return out


# ------------------------------------------------------------ build catalog --

def build(source_name, top_n=TOP_N):
    """Build the item catalog for one source, with SUBGROUP pre-filled where
    MIMIC metadata already answers the question."""
    source = SOURCES[source_name]
    d = load_dictionary(source)
    print(f"  {len(d)} items in {source.dictionary}")

    label_col = source.label_col
    d[label_col] = d[label_col].astype(str).str.strip()

    key_col = "SPEC_ITEMID" if source.join_col == "SPEC_ITEMID" else source.join_col
    if key_col not in d.columns:
        key_col = "ITEMID"
    top = compute_top_values(source, d[key_col].dropna().unique(), top_n=top_n)

    df = pd.DataFrame({
        "ITEMID": d["ITEMID"] if "ITEMID" in d.columns else pd.NA,
        "LABEL": d[label_col],
        "TOP_VALUES": [str(top.get(k, [])) for k in d[key_col]],
        "LINKSTO": source.linksto,
    })

    # SUBGROUP from MIMIC metadata where the dictionary already encodes it.
    if source_name == "labevents":
        df["SUBGROUP"] = (d["CATEGORY"].astype(str) + "(" + d["FLUID"].astype(str) + ")")
        df["GROUP"] = GROUP_LAB
    elif source_name == "microbiologyevents":
        # D_ITEMS.CATEGORY is SPECIMEN / ORGANISM / ANTIBACTERIUM here.
        category = d["CATEGORY"] if "CATEGORY" in d.columns else ""
        df["SUBGROUP"] = pd.Series(category, index=d.index).fillna("").astype(str)
        df["GROUP"] = df["SUBGROUP"].map(lambda s: source.default_group if s else "")
    else:
        df["SUBGROUP"] = ""
        df["GROUP"] = ""

    return df.drop_duplicates(subset=["LABEL"]).reset_index(drop=True)


def parse_top_values(cell):
    """TOP_VALUES round-trips through CSV as a repr string."""
    try:
        return ast.literal_eval(cell) if isinstance(cell, str) else cell
    except (ValueError, SyntaxError):
        return cell
