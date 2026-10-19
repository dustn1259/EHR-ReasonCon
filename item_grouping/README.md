# item_grouping — GROUP / SUBGROUP ontology for MIMIC-III items

Two ways to get the taxonomy:

- **Induce it** — `induce` discovers the GROUP/SUBGROUP tree from the item
  labels themselves. Nothing but MIMIC-III and an LLM. See
  [Inducing the ontology](#inducing-the-ontology-from-scratch).
- **Use the published one** — `taxonomy.py` holds the fixed taxonomy the paper
  used. `classify` assigns against it. Documented first, below.


Builds the **item catalog** that Stage-1's `cluster` NER method browses. Every
item recorded in MIMIC-III gets two labels:

| column | meaning |
|---|---|
| `SUBGROUP` | fine-grained clinical category, specific to the source table |
| `GROUP` | coarse category; `entity_extraction/ner_*.py` narrows GROUP first, then SUBGROUP |

Output schema matches `EHR_ReasonCon/patient_data/{hadm}/catalog_items.csv`:

```
LABEL  TOP_VALUES  SUBGROUP  GROUP  LINKSTO  ITEMID
```

`catalog_items.csv` is the per-patient slice of this catalog — this tool builds
the global version it is sliced from.

## How items get labelled

Three strategies, picked per source table (see `taxonomy.py: SOURCES`):

| strategy | source tables | how |
|---|---|---|
| `metadata` | labevents, microbiologyevents | MIMIC's dictionary already encodes it — `CATEGORY(FLUID)` for labs, `CATEGORY` for microbiology. No LLM call. |
| `rules` | procedureevents_mv | curated label → subgroup list (`PROCEDURE_RULES`); leftovers fall through to the LLM |
| `llm` | chartevents, outputevents, inputevents_cv/mv, prescriptions | batched LLM classification against a fixed subgroup list, then human review |

Anything the LLM does not resolve — blank, `Unknown`, a hallucinated subgroup,
or an item it silently dropped — is **never guessed**. It stays empty and lands
in the human review queue.

## Pipeline

```
build  ──→  classify  ──→  review  ──→  merge
 │            │              │            │
 │            │              │            └─ output/item_catalog.csv
 │            │              └─ human decides the blanks
 │            └─ rules → LLM (batched, cached)
 └─ enumerate items + collect TOP_VALUES from the event table
```

`TOP_VALUES` is the item's most frequent recorded values. It is shown to the
LLM and the reviewer as weak context only — the prompt explicitly says not to
classify on it, because a value list like `['80', '72', '95']` is equally
consistent with a heart rate and a blood glucose.

## Usage

```bash
conda activate llm-new
export MIMIC_DIR=/path/to/mimiciii
cd item_grouping

# 1. enumerate items + collect TOP_VALUES  (scans the big event tables)
python build_item_catalog.py build --source all

# 2. rules + LLM classification  (resumable; re-run continues where it stopped)
python build_item_catalog.py classify --source all

# 3. human review of whatever is left
python build_item_catalog.py review --source chartevents --export
#    ... edit the SUBGROUP column of output/review/chartevents_review.csv ...
python build_item_catalog.py review --source chartevents --apply

# 4. one catalog for all source tables
python build_item_catalog.py merge

python build_item_catalog.py status      # labelled / pending per source
```

`--source` takes any key of `taxonomy.SOURCES`, or `all`.

### Review options

| command | when to use |
|---|---|
| `review --export` / `--apply` | bulk review in a spreadsheet — export the queue, edit `SUBGROUP`, merge back |
| `review --interactive` | short queue; pick a subgroup by number in the terminal |
| `review --export --all-rows` | audit — export every row, not just the unlabelled ones |

Human decisions are recorded in `output/cache/{source}_human.json` and re-applied
on every later run, so **re-classifying never overwrites a human decision**.

### Running without credentials

```bash
python build_item_catalog.py classify --source all --skip-llm
```

Applies only the metadata and rule strategies; everything else goes to review.
Useful to check the catalog builds before spending tokens.

## Inducing the ontology from scratch

`induce` builds the taxonomy instead of assuming one. Given only the item
labels and their `TOP_VALUES`, the LLM proposes the categories itself.

```
GROUP induction          sweeps every item in the database
  seed         first batch  → propose an initial group set
  extend       each later batch → "do these items have a home? if several
               don't, propose a group" (existing groups are never renamed
               or split)
  settle       stop once N consecutive batches propose nothing new
  consolidate  one pass merging near-duplicate groups

assign groups            every item gets a group

SUBGROUP induction       same loop, run once per group over only that
                         group's items — so subgroups are discovered in the
                         context of their parent, not against the whole DB

assign subgroups         within each group, from that group's subgroups only
```

```bash
python build_item_catalog.py build --source all     # items must exist first
python build_item_catalog.py induce                 # groups, then subgroups
python build_item_catalog.py ontology               # print the result
python build_item_catalog.py assign                 # label every item with it
#   → output/ontology.json
#     output/item_catalog_induced.csv
```

`induce --stage groups` / `--stage subgroups` run one level at a time;
`--restart` discards `ontology.json` and starts over. `ontology.json` is plain
JSON — hand-edit it (rename a group, drop a bad subgroup) and re-run `assign`.

`assign --use-taxonomy` runs the same two-pass assignment against the fixed
`taxonomy.py` instead, which is a useful way to compare the two taxonomies on
identical items.

| variable | purpose | default |
|---|---|---|
| `INDUCE_SEED_BATCH` | items shown when proposing the initial set | 200 |
| `INDUCE_EXTEND_BATCH` | items per "anything missing?" batch | 150 |
| `INDUCE_PATIENCE` | quiet batches before the level is settled | 3 |
| `INDUCE_GROUP_MIN` / `_MAX` | target group count | 8 / 16 |
| `INDUCE_SUBGROUP_MIN` / `_MAX` | target subgroup count per group | 3 / 12 |

**Guard rails.** A category name is only accepted from a line that matches the
mandated `name | definition` format, so a conversational preamble never becomes
a group. A proposal that restates an existing category is dropped. During
assignment, only names on the current allowed list are written — a subgroup
belonging to a *different* group is rejected, and the item goes to review.
Caches are namespaced by a fingerprint of the category list, so switching
ontologies never reuses stale labels.

**Cost.** GROUP induction is one LLM call per `INDUCE_EXTEND_BATCH` items,
which for ~17k items is ~120 calls before the settle rule cuts it short.
Assignment is one call per `ITEM_GROUPING_BATCH_SIZE` items per level.

## Splitting the catalog per admission

Stage 1 never reads the global catalog. For each admission it reads two files
that partition it by whether the item is actually recorded for that patient:

| file | contents | used by |
|---|---|---|
| `patient_records.csv` | items the admission **has** | patient-specific extraction |
| `catalog_items.csv` | items the admission **lacks** (the complement) | ontology-guided extraction |

```bash
python build_item_catalog.py patient-catalog --patient-data /path/to/patient_data
```

`--patient-data` points at the directory of `{hadm}/` folders holding the
per-patient event tables. Both files are written next to those tables unless
`--dest` is given; `--catalog` picks which catalog to split (the induced one if
present, else `item_catalog.csv`).

Validated against the 105 released admissions: for every nursing/physician
admission this reproduces the shipped `patient_records.csv` with nothing
missing. Two deliberate differences remain — see the module docstring in
`patient_split.py`:

- drug matching is case/whitespace-insensitive, so all spellings of a drug the
  patient has are removed from the "lacks" list (~3.3k rows across the 105);
- discharge admissions also carry `DIAGNOSES_ICD` / `PROCEDURE_ICD` items
  (~500 per admission — the reason a discharge catalog is ~35k rows against
  ~17k). Those two source tables are not in `taxonomy.SOURCES` yet.

## Environment

| variable | purpose |
|---|---|
| `MIMIC_DIR` | MIMIC-III v1.4 csv directory (default `<release>/mimiciii`, see the root README) |
| `GOOGLE_APPLICATION_CREDENTIALS`, `PROJECT_ID`, `LOCATION` | Gemini via Vertex AI — same as the rest of the repo |
| `GOOGLE_API_KEY` | fallback, Gemini Developer API |
| `ITEM_GROUPING_MODEL` | default `gemini-2.5-pro` |
| `ITEM_GROUPING_BATCH_SIZE` | items per LLM call, default `100` |

## Files

| file | role |
|---|---|
| `taxonomy.py` | the **fixed** taxonomy: subgroup lists, `SUBGROUP_TO_GROUP`, `PROCEDURE_RULES`, per-source `SOURCES` spec |
| `catalog.py` | enumerate items, compute `TOP_VALUES` (chunked — `CHARTEVENTS.csv` is ~33 GB) |
| `classify.py` | fixed-taxonomy path: prompt building, batching, response parsing, resumable cache |
| `review.py` | human review: CSV export/apply, interactive, decision store |
| `ontology.py` | the **induced** ontology: tree container, merge/rename, JSON persistence |
| `induce.py` | seed → extend → settle → consolidate loop, for both levels |
| `assign.py` | two-pass assignment (GROUP then SUBGROUP) against any ontology |
| `build_item_catalog.py` | CLI orchestrating every stage |
| `prompt/classify_items.txt` | fixed-taxonomy classification prompt |
| `prompt/induce_{seed,extend,consolidate}.txt` | ontology induction prompts |
| `prompt/assign_items.txt` | assignment prompt, shared by both levels |

## Notes on faithfulness to the original notebooks

This replaces a set of per-table Jupyter notebooks. Behaviour is preserved
except where the notebook version was unsafe:

- **Batching.** The notebooks sent up to 7,000 items in a single prompt. Here
  the default is 100 per call, so a truncated or reordered response costs one
  batch, not the whole table.
- **Chunked scanning.** Event tables are counted in 2M-row chunks instead of
  being loaded whole.
- **Pinned value dtypes.** `Source.value_dtype` fixes whether a value column is
  text or numeric. pandas infers dtypes *per chunk*, so `CHARTEVENTS.VALUE`
  would otherwise come back as `int` in one chunk and `str` in the next.
- **No silent guesses.** The notebooks kept whatever string the model returned.
  Here anything off the allowed subgroup list is dropped into the review queue.
- **`TOP_VALUES` ordering.** The notebooks applied `list(set(...))` after taking
  the top 10, which randomized order between runs. Here values stay in
  descending frequency order.
- **Chart medication grouping.** The chartevents subgroup
  `"Medications & Infusions"` maps to `GROUP` `"Medications, Fluids & Nutrition"`.
