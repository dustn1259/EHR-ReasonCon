"""
Split the global item catalog into the two per-patient files Stage 1 reads.

For admission H, the global catalog L is partitioned by whether the item is
actually recorded for that patient:

    patient_records.csv   L_H      items H has  -> patient-specific extraction
    catalog_items.csv     L \\ L_H  items H lacks -> ontology-guided extraction

`entity_extraction/ner_*.py` loads `catalog_items.csv` as `remain_data_df` and
browses its GROUP -> SUBGROUP tree to recover entities mentioned in the note but
absent from the patient's own records, so it must be the *complement*, not the
whole catalog.

Validated against the 105 released admissions: for every nursing/physician
admission this reproduces the shipped `patient_records.csv` with nothing
missing. Two deliberate differences remain.

  Drug name matching is case- and whitespace-insensitive. The catalog carries
  "FUROSEMIDE", "Furosemide" and "furosemide" as separate rows; the shipped
  split matched them exactly, so spellings the patient demonstrably has stayed
  in the "patient lacks this" list. Matching all spellings removes ~3.3k such
  rows across the 105 admissions.

  Discharge admissions additionally carry DIAGNOSES_ICD and PROCEDURE_ICD
  items (~500 rows per admission, and the reason a discharge catalog is ~35k
  rows against ~17k for the others). Those two source tables are not part of
  `taxonomy.SOURCES`, so this tool cannot yet produce them -- run it for
  nursing/physician, and keep the released catalogs for discharge.
"""
import os
import re

import pandas as pd

# per-patient event file -> (catalog LINKSTO, id column(s) in the event file)
#
# microbiologyevents contributes three kinds of catalog item -- SPECIMEN,
# ORGANISM and ANTIBACTERIUM -- keyed by three different id columns. Reading
# only SPEC_ITEMID would leave every organism the patient grew (e.g. "YEAST",
# "ENTEROCOCCUS SP.") in the "items this patient lacks" list.
EVENT_FILES = {
    "chartevents.csv":        ("chartevents", ["ITEMID"]),
    "outputevents.csv":       ("outputevents", ["ITEMID"]),
    "inputevents_cv.csv":     ("inputevents_cv", ["ITEMID"]),
    "inputevents_mv.csv":     ("inputevents_mv", ["ITEMID"]),
    "procedureevents_mv.csv": ("procedureevents_mv", ["ITEMID"]),
    "labevents.csv":          ("Labevents", ["ITEMID"]),
    "microbiologyevents.csv": ("microbiologyevents",
                               ["SPEC_ITEMID", "ORG_ITEMID", "AB_ITEMID"]),
    "prescriptions.csv":      ("Prescriptions", ["DRUG"]),
}

OUT_COLS = ["LABEL", "TOP_VALUES", "SUBGROUP", "GROUP", "LINKSTO", "ITEMID"]


def _norm_drug(s):
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def patient_item_keys(patient_dir):
    """(itemids, drug_names) the admission actually has.

    ITEMIDs are pooled across source tables rather than matched per table.
    MIMIC item ids are globally unique (D_ITEMS, D_LABITEMS and the
    microbiology ranges do not overlap), but an item's dictionary LINKSTO does
    not always agree with where a given admission recorded it -- "Stool
    Estimate" (226586) is an outputevents item in D_ITEMS yet is charted in
    CHARTEVENTS for several admissions. Matching per table would leave such
    items in the "patient lacks this" list even though the patient has them.

    Prescriptions carry no item id, so drugs match on the normalized name.
    """
    itemids, drugs = set(), set()
    for filename, (_linksto, id_cols) in EVENT_FILES.items():
        path = os.path.join(patient_dir, filename)
        if not os.path.exists(path):
            continue
        for id_col in id_cols:
            try:
                col = pd.read_csv(path, usecols=[id_col], low_memory=False)[id_col].dropna()
            except (ValueError, KeyError):
                continue  # column absent in this export
            if id_col == "DRUG":
                drugs.update(_norm_drug(v) for v in col)
            else:
                itemids.update(int(v) for v in pd.to_numeric(col, errors="coerce").dropna())
    return itemids, drugs


def catalog_membership(catalog, itemids, drugs):
    """Boolean Series: does the admission have each row of the global catalog?"""
    flags = []
    for label, linksto, itemid in zip(catalog["LABEL"], catalog["LINKSTO"], catalog["ITEMID"]):
        if str(linksto) == "Prescriptions":
            flags.append(_norm_drug(label) in drugs)
        else:
            try:
                flags.append(int(float(itemid)) in itemids)
            except (TypeError, ValueError):
                flags.append(False)
    return pd.Series(flags, index=catalog.index)


def split_one(catalog, patient_dir, dest_dir=None, write=True):
    """Write patient_records.csv + catalog_items.csv for one admission.

    Returns (n_has, n_lacks).
    """
    dest_dir = dest_dir or patient_dir
    itemids, drugs = patient_item_keys(patient_dir)
    mask = catalog_membership(catalog, itemids, drugs)

    records = catalog[mask][OUT_COLS]
    remaining = catalog[~mask][OUT_COLS]
    if write:
        os.makedirs(dest_dir, exist_ok=True)
        records.to_csv(os.path.join(dest_dir, "patient_records.csv"), index=False)
        remaining.to_csv(os.path.join(dest_dir, "catalog_items.csv"), index=False)
    return len(records), len(remaining)


def split_all(catalog, patient_data_root, dest_root=None, log=print):
    """Run `split_one` for every {hadm}/ directory under `patient_data_root`."""
    if not os.path.isdir(patient_data_root):
        raise SystemExit(f"{patient_data_root} not found")

    hadms = sorted(d for d in os.listdir(patient_data_root)
                   if d.isdigit() and os.path.isdir(os.path.join(patient_data_root, d)))
    if not hadms:
        raise SystemExit(f"no {{hadm}}/ directories under {patient_data_root}")

    log(f"{len(hadms)} admission(s) under {patient_data_root}")
    done = 0
    for hadm in hadms:
        src = os.path.join(patient_data_root, hadm)
        dst = os.path.join(dest_root, hadm) if dest_root else src
        n_has, n_lacks = split_one(catalog, src, dst)
        if n_has == 0:
            log(f"  [WARN] {hadm}: no items matched -- are the event tables present?")
        done += 1
        if done % 20 == 0 or done == len(hadms):
            log(f"  {done}/{len(hadms)} done (last: {hadm} -> {n_has} records / {n_lacks} catalog)")
    return done
