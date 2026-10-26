# EHR-INSPECTOR

Official code for **EHR-INSPECTOR**, the consistency-verification framework introduced in
*Towards Error-Free EHRs: Reasoning-Intensive Consistency Verification Between Clinical Notes
and Structured Tables in Electronic Health Records*
([arXiv:2605.26463](https://arxiv.org/abs/2605.26463)), together with the tooling needed to
run it on the **EHR-ReasonCon** benchmark
([PhysioNet](https://physionet.org/content/ehr-reasoncon/1.0.0/)).

The repository contains the full three-stage EHR-INSPECTOR pipeline (anchor-entity and
temporal extraction, tool-assisted table exploration, consistency verification) and the
LLM-as-a-judge evaluation. The baseline and ablation variants reported in the paper are
not part of this repository.

## Layout

```
release/
├── README.md · CITATION.cff · croissant.json · requirements.txt · .env.example · .gitignore
├── smoke_test.py                    # credential-free checks (compile / secrets / dataset layout)
├── check_dataset.py                 # verify the downloaded EHR-ReasonCon bundle is complete and in place
├── build_patient_data.py            # (maintainers) assemble patient_data/ from the raw source dirs
├── entity_extraction/               # Stage 1: note segmentation + anchor/temporal NER
│   ├── ner_discharge.py   # discharge (main)
│   ├── ner_physician.py       # physician (main)
│   ├── ner_nursing.py           # nursing  (main)
│   ├── models.py                            # Vertex/Azure LLM clients
│   ├── requirements.txt                     # full conda-list freeze (reference; use root requirements.txt to install)
│   ├── utils/                               # utils, segmentation_utils, post_process,
│   │   │                                    #   time_post_process, ner_prompt_generator,
│   │   │                                    #   bm25_search_function
│   │   └── asset/                           # config_{discharge,physician,nursing}.yaml
│   ├── prompt/                              # segmentation / NER / temporal prompts
│   └── output/                              # Stage-1 pickles land here (gitignored)
│
├── tool_calling/                    # Stage 2 & 3: tool-augmented consistency agent + eval
│   ├── final_set_of_abbreviation1.csv       # shared lexical asset
│   ├── predefined_mapping.json              # shared item mapping
│   ├── current_hadm_id.txt                  # run-coordination file (rewritten at runtime; gitignored)
│   └── agent/
│       ├── tool_calling.py                  # main agent loop (Stage 2 & 3)
│       ├── tool_env.py                      # TerminalSearchTool (table-exploration tools)
│       ├── item_search_tool.py              # lexical/semantic item search
│       ├── data_show.py                     # table rendering helpers
│       ├── models.py                        # LLM clients (used by evaluation.py)
│       ├── evaluation.py                    # LLM-as-a-judge (harsh/lenient)
│       ├── prompt/                          # tool-calling / verification / eval-judge prompts
│       └── result/                          # prediction JSONs land here (gitignored)
│
├── item_grouping/                   # (optional, offline) builds the GROUP/SUBGROUP item catalog Stage 1 browses
│   ├── build_item_catalog.py            # CLI: build · classify · review · merge · induce · assign · patient-catalog
│   ├── taxonomy.py · catalog.py · classify.py · review.py · ontology.py · induce.py · assign.py · patient_split.py
│   ├── prompt/                          # classification / ontology-induction prompts
│   └── README.md                        # full documentation
├── mimiciii/                        # MIMIC-III v1.4 tables (PhysioNet; gitignored) — default location; or $MIMIC_DIR
│                                    #   the 8 gunzipped CSVs listed in Setup › 2b
└── EHR_ReasonCon/                   # the PhysioNet bundle (gitignored) — default location; or $EHR_REASONCON_DIR
    ├── annotations.csv · README.md      #   flat view of all annotations · dataset file guide
    ├── {note_type}/{split}/             #   {note_type}_{val,test}.csv  (notes)
    │                                    #   EHRReasonCon_{hadm}.0_data.json (gold labels)
    ├── patient_data/{hadm}/             #   per-patient EHR (integer hadm):
    │                                    #     patient_records.csv  catalog_items.csv
    │                                    #     chartevents.csv  labevents.csv  … (event tables)
    └── resources/                       #   MIMIC-derived tool assets (Stage 2):
                                         #     preprocessed_d_items.csv  top_values.json
```

The `entity_extraction/` ↔ `tool_calling/` **relative layout is intentional** — Stage 2
reads Stage-1 outputs via `../../entity_extraction/output/`, so keep the two code dirs as
siblings. The dataset is looked up through one setting, `EHR_REASONCON_DIR`
(default `<release>/EHR_ReasonCon`), so it may live anywhere.

## Environment

All scripts target **Python 3.10**. Install dependencies with:

```bash
pip install -r requirements.txt
```

(`entity_extraction/requirements.txt` is the full `conda list` freeze of the original
environment, kept only for reference.)

Quick sanity check without any credentials (T2 needs the dataset, see Setup › 2):

```bash
python smoke_test.py          # T0 compile · T1 secret scan · T2 EHR-ReasonCon layout · T3 MIMIC-III tables (warn only)
```

## Setup

### 1. Credentials (`.env`)

`cp .env.example .env` inside **both** run dirs (`entity_extraction/` and
`tool_calling/agent/`) and fill in the values. `GOOGLE_APPLICATION_CREDENTIALS`
is resolved **relative to the run dir**, so place the service-account JSON at
`entity_extraction/utils/asset/…json` and `tool_calling/agent/…json` (both
paths are gitignored). Supply your own keys — no credentials are included in this repository.

| Variable | Used for | Required |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS`, `PROJECT_ID`, `LOCATION` | Gemini via Vertex AI | yes |
| `AZURE_API_KEY`, `AZURE_API_VERSION`, `AZURE_ENDPOINT`, `AZURE_DEPLOYMENT` | Azure OpenAI — the temporal summary step of Stage 1 for discharge notes (`ner_discharge.py`) | discharge notes only |
| `EHR_REASONCON_DIR` | EHR-ReasonCon bundle location (all stages, `smoke_test.py`) | optional — defaults to `<release>/EHR_ReasonCon` |
| `MIMIC_DIR` | MIMIC-III v1.4 CSV folder (Stage 2) | optional — defaults to `<release>/mimiciii` |
| `EHRCON_DATASET_ROOT`, `EHRCON_PATIENT_DATA_ROOT` | only `build_patient_data.py` (raw source dirs) | not needed at run time |

### 2. Data — two PhysioNet downloads

The pipeline needs **two** credentialed PhysioNet datasets, neither of which is in this
repository. Download both yourself and place them under the release root (or point the
env vars at them). Both need a credentialed PhysioNet account, the CITI "Data or Specimens
Only Research" training, and the signed DUA of each project.

| | Dataset | Put it at | Override |
|---|---|---|---|
| 2a | **EHR-ReasonCon v1.0.0** — notes, gold labels, per-admission tables | `<release>/EHR_ReasonCon/` | `EHR_REASONCON_DIR` |
| 2b | **MIMIC-III v1.4** — 6 corpus-wide dictionary / ICD tables | `<release>/mimiciii/` | `MIMIC_DIR` |

#### 2a. EHR-ReasonCon (the benchmark bundle)

**EHR-ReasonCon v1.0.0** is published on PhysioNet under the Credentialed Health Data Use
Agreement.

> **Dataset:** https://physionet.org/content/ehr-reasoncon/1.0.0/ · DOI
> [10.13026/7m91-3010](https://doi.org/10.13026/7m91-3010)
> Requires a credentialed PhysioNet account, the CITI "Data or Specimens Only Research"
> training, and the signed DUA for this project.

**Download it yourself** from the project page: log in → *Files* → `EHR_ReasonCon.zip`
(≈0.5 GB; `LICENSE.txt` and `SHA256SUMS.txt` sit next to it). Verify the download, unpack it,
and place the resulting `EHR_ReasonCon/` folder at **`<release>/EHR_ReasonCon/`**:

```bash
sha256sum EHR_ReasonCon.zip   # b29efe69218e73d563b65a94a41141b0945ed9e5fa79633ec7f93bd4ad2a0c1b
unzip EHR_ReasonCon.zip -d <release>/
```

```
release/
├── entity_extraction/
├── tool_calling/
└── EHR_ReasonCon/          ← put the downloaded dataset here
    ├── discharge/  physician/  nursing/
    ├── patient_data/
    ├── resources/
    └── annotations.csv  README.md  …
```

Watch out for a double folder after unzipping (`EHR_ReasonCon/EHR_ReasonCon/…`): the five
sub-folders must sit *directly* inside `EHR_ReasonCon/`.
If you keep the dataset elsewhere (e.g. shared storage), either symlink it to that path or
export `EHR_REASONCON_DIR=/path/to/EHR_ReasonCon` before running any stage.

What the pipeline reads from the bundle (`EHR_ReasonCon/README.md` inside it documents every file):

- `{note_type}/{split}/` — notes (`{note_type}_{val,test}.csv`) + gold labels (`EHRReasonCon_{hadm}.0_data.json`)
- `patient_data/{hadm}/` — per-patient EHR (`{hadm}` = **integer** id):
  `patient_records.csv`, `catalog_items.csv` (Stage-1 entity extraction),
  `chartevents.csv`, `labevents.csv`, `prescriptions.csv`, … (Stage-2 event tables; also the
  source for the auto-built per-patient SQLite DBs below)
- `resources/` — Stage-2 tool assets: `preprocessed_d_items.csv` (D_ITEMS item dictionary),
  `d_labitems.csv` (D_LABITEMS dictionary), `top_values.json` (per-item value vocabulary)

#### 2b. MIMIC-III v1.4 (corpus-wide tables)

The bundle holds only the 105 admissions' rows. Entity→item alignment and ICD lookup also
need a few **corpus-wide** MIMIC-III tables, which PhysioNet does not allow us to
redistribute, so download them from the MIMIC-III project yourself:

> **Dataset:** MIMIC-III Clinical Database v1.4 — https://physionet.org/content/mimiciii/1.4/
> (credentialed; separate DUA from EHR-ReasonCon)

Only these **6 tables** are read. On PhysioNet they are gzipped (`*.csv.gz`); the code opens
plain `*.csv`, so gunzip them and place them at **`<release>/mimiciii/`**:

```
release/
└── mimiciii/
    ├── ADMISSIONS.csv          ├── D_ICD_DIAGNOSES.csv
    ├── D_ITEMS.csv             └── D_ICD_PROCEDURES.csv
    ├── D_LABITEMS.csv
    └── PRESCRIPTIONS.csv
```

```bash
# e.g. from the PhysioNet "Files" page, download just these 6 *.csv.gz into release/mimiciii/, then
cd mimiciii && gunzip *.csv.gz
```

| Table | Used by |
|---|---|
| `D_ICD_DIAGNOSES.csv`, `D_ICD_PROCEDURES.csv` | Stage 2 — ICD title lookup (the per-admission ICD codes themselves ship in the bundle) |
| `D_LABITEMS.csv`, `D_ITEMS.csv`, `PRESCRIPTIONS.csv`, `ADMISSIONS.csv` | Stage 2 (`item_search_tool.py`, `tool_env.py`) |

Stage 1 reads no MIMIC-III tables; the catalog slices it needs are part of the bundle.

If you already have a full MIMIC-III v1.4 CSV folder elsewhere, do not copy it — export
`MIMIC_DIR=/path/to/mimiciii` instead (the other tables in it are simply ignored).

#### Check both

```bash
python check_dataset.py            # 1. EHR-ReasonCon layout  2. MIMIC-III tables — prints exactly what is missing
```

`smoke_test.py` runs the same checks (MIMIC-III as a warning, since it is only needed at run time).

> Maintainers only: `build_patient_data.py` re-assembles `patient_data/` from the raw
> (pre-release) source dirs given `EHRCON_DATASET_ROOT` / `EHRCON_PATIENT_DATA_ROOT`.
> End users never need it — the bundle already contains `patient_data/`.

**Per-patient SQLite DBs (automatic).** Stage 2's `Item_Search` tool queries one SQLite
database per admission (`tool_calling/new_final_db/sample_{hadm}.0_a_p.db`, 14 tables). They are
MIMIC-derived, gitignored, and **built automatically on first use** from the bundle's
`patient_data/` + the `$MIMIC_DIR` dictionaries (`ensure_patient_db` in
`item_search_tool.py`) — no setup step needed, ~400 MB total once all notes have run.

## Run

**All three stages are controlled by environment variables — no code or config edits.**
Set them once with `export`, then run the stages in order.

| Variable | Meaning | Default | Read by |
|---|---|---|---|
| `NOTE_IDX` | note index within the split; **unset = all notes** | (all) | Stage 1, Stage 2, Eval |
| `TEST_VAL` | split: `val` or `test` | `val` | Stage 1, Stage 2, Eval |
| `NOTE_TYPE` | `discharge` / `physician` / `nursing` | `discharge` | Stage 2, Eval |
| `NER_METHOD` | Entity-extraction tag used in Stage-1/2 output file names | `cluster` | Stage 2, Eval |
| `MODEL_NAME` | backbone tag in the Stage-1/2 file names; Stage 1 always writes `gemini`, so keep the default | `gemini` | Stage 2, Eval |
| `DATA_VERSION` | version tag in the Stage-2 output name; Stage-1 files are always `_v2_`, so keep the default | `v2` | Stage 2, Eval |
| `EVAL_MODE` | judge strictness: `harsh` or `lenient` | `harsh` | Eval |
| `VERIFICATION_RETRIES` | extra attempts for the final verification call when its answer is unparsable; after that the entity is recorded as `verification failed` | `2` | Stage 2 |
| `ITEM_SEARCH_ABSENT_LIMIT` | max "not in patient record" candidates shown by `Item_Search` | `100` | Stage 2 |

For **Stage 1**, the note type is fixed by *which script you run* — it does not read
`NOTE_TYPE`. Pick the script to match `NOTE_TYPE`:

| NOTE_TYPE | Stage-1 script |
|---|---|
| `discharge` | `ner_discharge.py` |
| `physician` | `ner_physician.py` |
| `nursing`   | `ner_nursing.py` |

### Example — discharge / valid, note index 8

Measured on this note (`HADM_ID 139574`): Recall 74.3 / Precision 60.9 (harsh).

> These numbers are illustrative. The pipeline calls hosted LLMs (Gemini via Vertex AI,
> optionally Azure OpenAI), so exact scores vary with the model version/snapshot and
> sampling — treat them as a sanity check, not a fixed benchmark.

```bash
export NOTE_TYPE=discharge TEST_VAL=val NOTE_IDX=8    # keep these consistent across all 3 stages

# ── Stage 1: segmentation + anchor/temporal NER ───────────────────────────────
cd entity_extraction
python ner_discharge.py utils/asset/config_discharge.yaml   # config = prompt paths only (MIMIC tables via MIMIC_DIR env)
#   → output/clusterner_time_result_discharge_val_8_v2_gemini_2.pkl
#     output/subtext_divided_discharge_val_8_gemini_2.pkl
#     ../tool_calling/current_hadm_id.txt

# ── Stage 2 & 3: tool-augmented consistency check ─────────────────────────────
cd ../tool_calling/agent                     # per-patient DBs auto-build into ../new_final_db/ on first use
python tool_calling.py
#   → result/concheck_result_8_discharge_val_cluster_v2_2.json

# ── Evaluation: LLM-as-a-judge (Recall / Precision) ───────────────────────────
python evaluation.py           # add EVAL_MODE=lenient to re-score leniently (no re-run of stages 1–2)
#   → prints  RECALL: …  PRECISION: …
```

### Other examples

```bash
# All physician TEST notes, end to end:
export NOTE_TYPE=physician TEST_VAL=test ; unset NOTE_IDX
python ner_physician.py utils/asset/config_physician.yaml   # Stage 1
# … then Stage 2 (tool_calling.py) and Eval (evaluation.py) as above

# One nursing valid note (index 3):
export NOTE_TYPE=nursing TEST_VAL=val NOTE_IDX=3
python ner_nursing.py utils/asset/config_nursing.yaml           # Stage 1
```

**Stage-to-stage naming must line up.** Stage 2 loads Stage-1 outputs by exact filename:
`{NER_METHOD}ner_time_result_{NOTE_TYPE}_{TEST_VAL}_{id}_v2_{MODEL_NAME}_2.pkl`
and `subtext_divided_{NOTE_TYPE}_{TEST_VAL}_{id}_{MODEL_NAME}_2.pkl` — so the `NOTE_TYPE` /
`TEST_VAL` / `NER_METHOD` / `MODEL_NAME` env vars must be identical across the stages (and the
Stage-1 script must match `NOTE_TYPE`). A mismatch is silently skipped. The config `.yaml`
passed to Stage 1 only carries prompt locations; it does not select the note. (No Stage-1 script
writes a `d_p_concheck` file; Stage 2 treats it as empty.)

## Optional: rebuilding the item catalog (`item_grouping/`)

Stage 1 browses a GROUP → SUBGROUP catalog of MIMIC-III items; the
per-admission slices of that catalog (`patient_records.csv` / `catalog_items.csv`) already
ship inside the PhysioNet bundle, so **nothing here is needed to run the pipeline**.
`item_grouping/` is the tooling that produced them, kept for transparency and for anyone
who wants to re-derive or extend the taxonomy:

- `build_item_catalog.py classify` labels every MIMIC-III item with the paper's fixed taxonomy
  (`taxonomy.py`) using metadata rules + batched LLM calls + a human review queue;
- `build_item_catalog.py induce` instead discovers the GROUP/SUBGROUP tree from the item
  labels alone;
- `build_item_catalog.py patient-catalog` re-splits a catalog into the two per-admission files.

It reads the full MIMIC-III event tables from `MIMIC_DIR` (not just the 6 tables above) and
calls Gemini with the same credentials as the rest of the repo. See `item_grouping/README.md`.

## Not included in this repository

- **All credential JSON files** (service-account keys, API keys).
- Baseline / ablation scripts, notebooks, scratch/`old_*` dirs.
- Large regenerable artifacts: per-patient SQLite DBs (auto-built on first use),
  eval logs, raw annotator label dirs (`v2_labeling_*`).

## Dataset metadata (Croissant)

`croissant.json` describes the EHR-ReasonCon dataset in the
[MLCommons Croissant 1.0](https://mlcommons.org/croissant/) format (JSON-LD): the PhysioNet
archive as the single distribution, every file family inside it (notes, annotation JSONs,
`annotations.csv`, the 12 per-admission tables, item dictionaries), and a record set with
typed fields for each table. It is generated from the published bundle's own headers and
counts. The data itself is **not** in this repo; `contentUrl` points at the archive on
PhysioNet (`EHR_ReasonCon.zip`, credentialed access) and carries its published sha256. Validate with
`pip install mlcroissant && python -c "import mlcroissant as mlc; mlc.Dataset('croissant.json')"`.

## Data use

The **EHR-ReasonCon dataset** (https://physionet.org/content/ehr-reasoncon/1.0.0/,
DOI 10.13026/7m91-3010) is derived from MIMIC-III and is distributed only via PhysioNet under
the PhysioNet Credentialed Health Data Use Agreement — it is **not** included in this
repository. Do not redistribute the downloaded bundle, the auto-built SQLite DBs, or run
outputs that embed patient data.

## Citation

If you use this code or the EHR-ReasonCon benchmark, please cite the paper **and** the
PhysioNet dataset record (machine-readable version in `CITATION.cff`):

- Paper: https://arxiv.org/abs/2605.26463
- Dataset: https://physionet.org/content/ehr-reasoncon/1.0.0/ (DOI 10.13026/7m91-3010)

```bibtex
@article{kwon2026ehrreasoncon,
  title   = {Towards Error-Free EHRs: Reasoning-Intensive Consistency Verification
             Between Clinical Notes and Structured Tables in Electronic Health Records},
  author  = {Kwon, Yeonsu and Kim, Jiho and Choi, Junseong and Rabaey, Paloma and
             Kim, Minseo and Im, Sujeong and Yang, Jeewon and Lee, Jun-Min and
             Lee, Sangji and Kim, Jiwon and Yoon, Hangyul and Kwon, Hyunwook and Choi, Edward},
  journal = {arXiv preprint arXiv:2605.26463},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.26463}
}

@misc{kwon2026ehrreasoncon_physionet,
  title     = {EHR-ReasonCon: Reasoning-Intensive Consistency Verification Between
               Clinical Notes and Structured Tables in Electronic Health Records (version 1.0.0)},
  author    = {Kwon, Yeonsu and Kim, Jiho and Choi, Junseong and Rabaey, Paloma and
               Kim, Minseo and Im, Sujeong and Yang, Jeewon and Lee, Jun-Min and
               Lee, Sangji and Kim, Jiwon and Yoon, Hangyul and Kwon, Hyunwook and Choi, Edward},
  year      = {2026},
  publisher = {PhysioNet},
  doi       = {10.13026/7m91-3010},
  url       = {https://physionet.org/content/ehr-reasoncon/1.0.0/}
}
```

PhysioNet also asks that you cite the platform: Pollard, T., Moody, B. E., Lehman, L., Gow, B.,
Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). *PhysioNet as a global
platform for biomedical research.* Nature Health. https://doi.org/10.1038/s44360-026-00096-z
