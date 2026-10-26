#!/usr/bin/env python3
"""
Verify that the EHR-ReasonCon bundle is in place and complete.

Get the data yourself from PhysioNet (credentialed access required):

    https://physionet.org/content/ehr-reasoncon/1.0.0/      DOI 10.13026/7m91-3010

Download EHR_ReasonCon.zip from the project's Files section (sha256
b29efe69218e73d563b65a94a41141b0945ed9e5fa79633ec7f93bd4ad2a0c1b), unpack it, and put
the resulting EHR_ReasonCon/ folder (containing  discharge/  physician/  nursing/
patient_data/  resources/) at

    <release>/EHR_ReasonCon/            (default; nothing else to configure)

or anywhere else, in which case export  EHR_REASONCON_DIR=/that/folder  before running
any stage. Then run:

    python check_dataset.py             # or:  python smoke_test.py

It cross-checks notes CSV <-> gold-label JSON <-> patient_data/{hadm}/ (12 tables) and
the three resources/ files, and prints exactly what is missing if anything is.

It also checks MIMIC-III v1.4 (a separate credentialed PhysioNet dataset,
https://physionet.org/content/mimiciii/1.4/): the six tables Stage 2 reads must be
present, gunzipped, at  <release>/mimiciii/  or at $MIMIC_DIR.
"""
import argparse
import glob
import json
import os
import sys

RELEASE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.join(RELEASE_ROOT, "EHR_ReasonCon")
PHYSIONET_PAGE_URL = "https://physionet.org/content/ehr-reasoncon/1.0.0/"
DEFAULT_MIMIC_DIR = os.path.join(RELEASE_ROOT, "mimiciii")
MIMIC_PAGE_URL = "https://physionet.org/content/mimiciii/1.4/"
MIMIC_TABLES = ["D_LABITEMS.csv", "D_ITEMS.csv", "PRESCRIPTIONS.csv", "D_ICD_DIAGNOSES.csv",
                "D_ICD_PROCEDURES.csv", "ADMISSIONS.csv"]

NOTE_TYPES = ["discharge", "physician", "nursing"]
SPLITS = [("valid", "val"), ("test", "test")]
PATIENT_TABLES = [
    "chartevents.csv", "labevents.csv", "prescriptions.csv", "inputevents_cv.csv",
    "inputevents_mv.csv", "microbiologyevents.csv", "procedureevents_mv.csv",
    "outputevents.csv", "diagnosis_icd.csv", "procedure_icd.csv",
    "patient_records.csv", "catalog_items.csv",
]
RESOURCE_FILES = ["preprocessed_d_items.csv", "d_labitems.csv", "top_values.json"]


def dataset_dir():
    return os.getenv("EHR_REASONCON_DIR", DEFAULT_DEST).rstrip("/")


def mimic_dir():
    return os.getenv("MIMIC_DIR", DEFAULT_MIMIC_DIR).rstrip("/")


def validate_mimic(root, printer=print):
    """Check the MIMIC-III v1.4 tables the pipeline reads. PhysioNet ships them as *.csv.gz;
    the code opens *.csv, so a gzipped-only table gets its own hint.
    Returns the number of tables that are not usable."""
    if not os.path.isdir(root):
        printer(f"  [FAIL] MIMIC-III dir  — {root} does not exist. Download MIMIC-III v1.4 from {MIMIC_PAGE_URL} "
                f"(credentialed), gunzip the tables, and place them there — or set MIMIC_DIR to where they are.")
        return len(MIMIC_TABLES)
    missing, gz_only = [], []
    for t in MIMIC_TABLES:
        if os.path.exists(os.path.join(root, t)):
            continue
        (gz_only if os.path.exists(os.path.join(root, t + ".gz")) else missing).append(t)
    ok = not (missing or gz_only)
    detail = f"all {len(MIMIC_TABLES)} tables present" if ok else ""
    if gz_only:
        detail += f"still gzipped: {gz_only} — run `gunzip {root}/*.csv.gz`"
    if missing:
        detail += (" | " if detail else "") + f"missing: {missing} — download from {MIMIC_PAGE_URL}"
    printer(f"  [{'OK ' if ok else 'FAIL'}] MIMIC-III tables in {root}  — {detail}")
    return len(missing) + len(gz_only)


def label_path(split_dir, hadm):
    """Gold-label JSON for one note. The shipped name is built from the CSV's HADM_ID,
    which pandas reads as float ("112673.0"); accept the integer spelling too."""
    for stem in (str(hadm), str(int(float(hadm))), f"{int(float(hadm))}.0"):
        p = os.path.join(split_dir, f"EHRReasonCon_{stem}_data.json")
        if os.path.exists(p):
            return p
    return None


def looks_like_dataset_root(d):
    return (os.path.isdir(os.path.join(d, "patient_data"))
            and os.path.isdir(os.path.join(d, "resources"))
            and all(os.path.isdir(os.path.join(d, nt)) for nt in NOTE_TYPES))


def find_misplaced_root(d, max_depth=3):
    """If the user unpacked the archive with its wrapper folder (e.g. EHR_ReasonCon/EHR_ReasonCon/
    or ehr-reasoncon-1.0.0/...), point at the real root so the fix is obvious."""
    d = os.path.abspath(d)
    for cur, dirs, _ in os.walk(d, followlinks=True):
        if cur[len(d):].count(os.sep) >= max_depth:
            dirs[:] = []
            continue
        dirs[:] = [x for x in dirs if x not in ("patient_data", ".git", "__pycache__")]
        if cur != d and looks_like_dataset_root(cur):
            return cur
    return None


def validate_dataset(root, printer=print):
    """Returns the number of failed checks (0 = everything the pipeline needs is present)."""
    import pandas as pd  # local import so smoke_test can report a missing pandas itself

    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        printer(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            failures += 1

    if not os.path.isdir(root):
        check("dataset present", False,
              f"{root} does not exist. Download EHR-ReasonCon from {PHYSIONET_PAGE_URL} and place it "
              "there, or set EHR_REASONCON_DIR to where you put it.")
        return failures

    if not looks_like_dataset_root(root):
        inner = find_misplaced_root(root)
        hint = (f"the dataset seems to be one level down at {inner} — move its contents up, "
                f"or set EHR_REASONCON_DIR={inner}") if inner else \
               "expected discharge/ physician/ nursing/ patient_data/ resources/ directly inside it"
        check("dataset layout", False, f"{root}: {hint}")
        return failures

    for nt in NOTE_TYPES:
        for split_dir, split_tag in SPLITS:
            sdir = os.path.join(root, nt, split_dir)
            csv = os.path.join(sdir, f"{nt}_{split_tag}.csv")
            if not os.path.exists(csv):
                check(f"{nt}/{split_dir}", False, f"missing {os.path.relpath(csv, root)}")
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as e:  # noqa: BLE001
                check(f"{nt}/{split_dir}", False, f"cannot read notes CSV: {e!r}")
                continue
            n_labels = len(glob.glob(os.path.join(sdir, "EHRReasonCon_*_data.json")))
            missing_lbl, bad_lbl, missing_pd, missing_tbl = [], [], [], []
            for hadm in df["HADM_ID"]:
                lp = label_path(sdir, hadm)
                if lp is None:
                    missing_lbl.append(hadm)
                else:
                    try:
                        json.load(open(lp, encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        bad_lbl.append(os.path.basename(lp))
                pdir = os.path.join(root, "patient_data", str(int(float(hadm))))
                if not os.path.isdir(pdir):
                    missing_pd.append(int(float(hadm)))
                else:
                    gone = [t for t in PATIENT_TABLES if not os.path.exists(os.path.join(pdir, t))]
                    if gone:
                        missing_tbl.append((int(float(hadm)), gone))
            ok = not (missing_lbl or bad_lbl or missing_pd or missing_tbl) and n_labels == len(df)
            detail = f"notes={len(df)} labels={n_labels}"
            if missing_lbl:
                detail += f" | no label for HADM {missing_lbl[:3]}{'…' if len(missing_lbl) > 3 else ''}"
            if bad_lbl:
                detail += f" | unreadable {bad_lbl[:3]}"
            if missing_pd:
                detail += f" | no patient_data for {missing_pd[:3]}{'…' if len(missing_pd) > 3 else ''}"
            if missing_tbl:
                h, gone = missing_tbl[0]
                detail += f" | patient_data/{h} lacks {gone}" + (f" (+{len(missing_tbl)-1} more)" if len(missing_tbl) > 1 else "")
            check(f"{nt}/{split_dir}", ok, detail)

    for rf in RESOURCE_FILES:
        p = os.path.join(root, "resources", rf)
        check(f"resources/{rf}", os.path.exists(p) and os.path.getsize(p) > 0)

    n_pd = len([d for d in glob.glob(os.path.join(root, "patient_data", "*")) if os.path.isdir(d)])
    check("patient_data folders", n_pd > 0, f"{n_pd} admissions")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None,
                    help="EHR-ReasonCon folder to check (default: $EHR_REASONCON_DIR, else <release>/EHR_ReasonCon)")
    ap.add_argument("--mimic-dir", default=None,
                    help="MIMIC-III v1.4 csv folder to check (default: $MIMIC_DIR, else <release>/mimiciii)")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.dir) if args.dir else dataset_dir()
    mimic = os.path.abspath(args.mimic_dir) if args.mimic_dir else mimic_dir()

    print(f"1. EHR-ReasonCon at {root}")
    fails = validate_dataset(root)
    print(f"2. MIMIC-III v1.4 at {mimic}")
    mimic_fails = validate_mimic(mimic)

    exports = []
    if os.path.realpath(root) != os.path.realpath(DEFAULT_DEST):
        exports.append(f"export EHR_REASONCON_DIR={root}")
    if os.path.realpath(mimic) != os.path.realpath(DEFAULT_MIMIC_DIR):
        exports.append(f"export MIMIC_DIR={mimic}")

    if fails == 0 and mimic_fails == 0:
        print("\n=== both datasets ready ===")
        if exports:
            print("Non-default locations — run every stage with:\n    " + "\n    ".join(exports))
        return 0
    print(f"\n=== {fails + mimic_fails} problem(s) ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
