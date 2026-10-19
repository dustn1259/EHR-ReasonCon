"""
Credential-free smoke test for the EHR-INSPECTOR release.

Run from the repo root:  python smoke_test.py
Verifies the parts that do NOT need Vertex/Azure keys or MIMIC-III:
  T0. every .py compiles
  T1. no hardcoded secrets remain
  T2. the EHR-ReasonCon bundle (PhysioNet) is in place and internally consistent:
      notes CSV <-> gold-label JSON <-> patient_data/{hadm}/ (12 tables) <-> resources/
      Location: $EHR_REASONCON_DIR, else <release>/EHR_ReasonCon
      (download it yourself from https://physionet.org/content/ehr-reasoncon/1.0.0/ and
       place it there — see README "Setup > 2. Data")
  T3. the eight MIMIC-III v1.4 tables Stage 1/2 read are at <release>/mimiciii (or $MIMIC_DIR)
      (reported as WARN, not FAIL — the bundle alone is enough for T2)
"""
import os, sys, glob, py_compile, re

from check_dataset import validate_dataset, validate_mimic, mimic_dir, DEFAULT_DEST

ROOT = os.path.dirname(os.path.abspath(__file__))
EHR_REASONCON = os.getenv("EHR_REASONCON_DIR", DEFAULT_DEST).rstrip("/")
fail = 0

def check(name, ok, detail=""):
    global fail
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok: fail += 1

print("T0. py_compile all .py files")
py_files = [f for f in glob.glob(f"{ROOT}/**/*.py", recursive=True)
            if "/EHR_ReasonCon/" not in f]
bad = 0
for f in py_files:
    try:
        py_compile.compile(f, doraise=True)
    except py_compile.PyCompileError as e:
        bad += 1
        check(os.path.relpath(f, ROOT), False, str(e).splitlines()[-1])
if not bad:
    check("all files compiled", True, f"{len(py_files)} files")

print("T1. hardcoded-secret scan")
pat = re.compile(r"sk-[A-Za-z0-9]{20,}|AIzaSy[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16}")
hits = []
for f in py_files:
    for i, line in enumerate(open(f, encoding="utf-8", errors="ignore"), 1):
        if pat.search(line):
            hits.append(f"{os.path.relpath(f, ROOT)}:{i}")
check("no API keys in code", not hits, ", ".join(hits) if hits else "")

print(f"T2. EHR-ReasonCon bundle at {EHR_REASONCON}")
fail += validate_dataset(EHR_REASONCON)   # prints its own [OK]/[FAIL] lines, with a download hint if absent

print("T3. MIMIC-III v1.4 tables (needed at run time; download from https://physionet.org/content/mimiciii/1.4/)")
if validate_mimic(mimic_dir(), printer=lambda m: print(m.replace("[FAIL]", "[WARN]"))):
    print("      ^ not required for this test, but Stage 1 (discharge) and Stage 2 will not run without them")

print(f"\n=== {'ALL PASS' if fail == 0 else str(fail) + ' FAIL'} ===")
sys.exit(1 if fail else 0)
