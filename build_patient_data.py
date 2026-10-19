"""
Assemble per-patient EHR data into EHR_ReasonCon/patient_data/{hadm}/ with intuitive names.

Source data (MIMIC-derived, credentialed — kept out of GitHub, uploaded to PhysioNet):
  - Stage-1 record profiles:  {EHRCON_DATASET_ROOT}/personal_csv/{hadm}_folder/
        discharge:          true_final_record_s_{hadm}.csv , final_remain_record_s_{hadm}.csv
        physician/nursing:  fianl_record_{hadm}.csv        , remain_record_{hadm}.csv
  - Stage-2 event tables:     {EHRCON_PATIENT_DATA_ROOT}/{hadm}/{table}_{hadm}.csv

Destination (single, note-type-agnostic, per integer HADM id):
  EHR_ReasonCon/patient_data/{hadm}/
        patient_records.csv   catalog_items.csv
        chartevents.csv  labevents.csv  prescriptions.csv  inputevents_mv.csv
        inputevents_cv.csv  microbiologyevents.csv  procedureevents_mv.csv
        outputevents.csv  diagnosis_icd.csv  procedure_icd.csv

Run once from the release root:  python build_patient_data.py
(writes into ./EHR_ReasonCon/patient_data/, sibling of entity_extraction/ and tool_calling/)
"""
import os, glob, shutil, re

ROOT = os.path.dirname(os.path.abspath(__file__))
EHR_REASONCON = os.getenv("EHR_REASONCON_DIR", os.path.join(ROOT, "EHR_ReasonCon")).rstrip("/")
DEST = os.path.join(EHR_REASONCON, "patient_data")

_dataset_root = os.getenv("EHRCON_DATASET_ROOT")
_events_root  = os.getenv("EHRCON_PATIENT_DATA_ROOT")
if not _dataset_root or not _events_root:
    raise SystemExit("Set EHRCON_DATASET_ROOT and EHRCON_PATIENT_DATA_ROOT to the raw source dirs.")
SRC_PERSONAL = os.path.join(_dataset_root, "personal_csv")
SRC_EVENTS   = _events_root

# note_type -> (patient_records source stem, catalog_items source stem)
RECORD_SRC = {
    "discharge": ("true_final_record_s", "final_remain_record_s"),
    "physician": ("fianl_record", "remain_record"),
    "nursing":   ("fianl_record", "remain_record"),
}
EVENT_TABLES = ["chartevents", "labevents", "prescriptions", "inputevents_mv",
                "inputevents_cv", "microbiologyevents", "procedureevents_mv",
                "outputevents", "diagnosis_icd", "procedure_icd"]


def hadms_by_note_type():
    """Return {hadm_int: note_type} from the gold label JSON files in EHR_ReasonCon."""
    out = {}
    for nt in RECORD_SRC:
        for split in ("valid", "test"):
            for p in glob.glob(f"{EHR_REASONCON}/{nt}/{split}/EHRReasonCon_*_data.json"):
                m = re.search(r"EHRReasonCon_(\d+)", os.path.basename(p))
                if m:
                    out[int(m.group(1))] = nt
    return out


def copy_one(hadm, note_type):
    dst = os.path.join(DEST, str(hadm))
    os.makedirs(dst, exist_ok=True)
    missing = []
    rec_stem, cat_stem = RECORD_SRC[note_type]
    folder = os.path.join(SRC_PERSONAL, f"{hadm}_folder")
    pairs = [(f"{rec_stem}_{hadm}.csv", "patient_records.csv"),
             (f"{cat_stem}_{hadm}.csv", "catalog_items.csv")]
    for src_name, dst_name in pairs:
        src = os.path.join(folder, src_name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dst, dst_name))
        else:
            missing.append(src_name)
    for table in EVENT_TABLES:
        src = os.path.join(SRC_EVENTS, str(hadm), f"{table}_{hadm}.csv")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dst, f"{table}.csv"))
        else:
            missing.append(f"{table}_{hadm}.csv")
    return missing


def main():
    os.makedirs(DEST, exist_ok=True)
    mapping = hadms_by_note_type()
    print(f"{len(mapping)} labeled admissions found. Copying into {DEST}/ ...")
    ok, warned = 0, 0
    for hadm, nt in sorted(mapping.items()):
        missing = copy_one(hadm, nt)
        if missing:
            warned += 1
            print(f"  [WARN] {hadm} ({nt}): missing {len(missing)} file(s): {missing[:3]}{'...' if len(missing)>3 else ''}")
        else:
            ok += 1
    print(f"\nDone. complete={ok}  with_missing={warned}  total={len(mapping)}")


if __name__ == "__main__":
    main()
