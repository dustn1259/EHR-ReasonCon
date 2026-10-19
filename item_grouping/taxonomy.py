"""
GROUP / SUBGROUP taxonomy for MIMIC-III items.

Every item that appears in a patient's `catalog_items.csv` carries two labels:

  SUBGROUP  fine-grained clinical category (source-table specific)
  GROUP     coarse category the Stage-1 `cluster` NER method browses first
            (see `entity_extraction/ner_*.py`: GROUP is narrowed, then SUBGROUP)

Three assignment strategies are used, one per source table:

  METADATA  MIMIC's own dictionary columns already encode the category
            (labevents -> CATEGORY(FLUID), microbiologyevents -> CATEGORY).
  RULES     a curated label -> subgroup keyword map (procedureevents_mv).
  LLM       batched LLM classification against a fixed subgroup list,
            followed by human review (chartevents, outputevents,
            inputevents_cv/mv, prescriptions).
"""

# ---------------------------------------------------------------- groups ----

GROUP_MEDS = "Medications, Fluids & Nutrition"
GROUP_PROC = "Procedures / lines & Orders"
GROUP_OUTPUT = "Output and Removal"
GROUP_LAB = "Labs / ABG-Chemistry"
GROUP_MICRO = "microbiology"
GROUP_ADMIN = "Ambiguous, Administrative / Admission & Demographics"
GROUP_SKIN = "Skin / Wounds & Devices / Safety"

# ------------------------------------------------------------- subgroups ----

MED_SUBGROUPS = [
    "Fluids & Nutrition",
    "Cardio & Hematology",
    "Infectious Diseases & Immunology",
    "CNS, Pain & Sedation",
    "Endocrine, Renal & GI",
    "Respiratory, ENT/Ophth & Derm",
    "Formulations, Devices, StudyCodes & Ambiguous",
]

PROC_SUBGROUPS = [
    "Access & Lines",
    "Airway & Ventilation",
    "Renal & Dialysis",
    "Cardiac",
    "Diagnostics (Imaging/Labs/Cultures/Swabs/ECG·EEG)",
    "Care Flow, Comms , SafetyAdmin",
    "GI&IR",
    "Neuro & ICP",
]

CHART_SUBGROUPS = [
    "Respiratory / Ventilation & Gas Monitoring",
    "Hemodynamics & Vitals",
    "Labs / ABG-Chemistry",
    "Medications & Infusions",
    GROUP_PROC,
    GROUP_SKIN,
    "Neuro/Pain/Functional",
    GROUP_ADMIN,
    "allergy",
    "diagnosis",
]

OUTPUT_SUBGROUPS = [
    "dialysis_ultrafiltration",
    "urinary_output",
    "pleural_chest_tubes",
    "abdominal_paracentesis",
    "csf_drainage",
    "gi_tubes_output",
    "wound_local_drains",
    "bloodloss_apheresis_ops",
    "drain_devices",
    "bowel_ostomy_output",
    "emesis_oral_gastric",
    "hepatobiliary_pancreatic_drains",
    "Location-based output",
    "Cardiac output",
    "Adjustment/aggregated values",
    "Ambiguous values",
]

MICRO_SUBGROUPS = ["SPECIMEN", "ORGANISM", "ANTIBACTERIUM"]

# inputevents items are mostly drugs/fluids, but the CareVue free-form intake
# labels also cover procedures and administrative entries.
INPUT_SUBGROUPS = MED_SUBGROUPS + PROC_SUBGROUPS + [GROUP_ADMIN, "allergy", "diagnosis"]

UNKNOWN = "Unknown"

# --------------------------------------------------- subgroup -> group ------

# Chart subgroups double as group names.
SUBGROUP_TO_GROUP = {sg: sg for sg in CHART_SUBGROUPS}
SUBGROUP_TO_GROUP["Medications & Infusions"] = GROUP_MEDS
SUBGROUP_TO_GROUP.update({sg: GROUP_MEDS for sg in MED_SUBGROUPS})
SUBGROUP_TO_GROUP.update({sg: GROUP_PROC for sg in PROC_SUBGROUPS})
SUBGROUP_TO_GROUP.update({sg: GROUP_OUTPUT for sg in OUTPUT_SUBGROUPS})
SUBGROUP_TO_GROUP.update({sg: GROUP_MICRO for sg in MICRO_SUBGROUPS})


def group_for(subgroup, source):
    """Resolve GROUP from SUBGROUP. Lab subgroups are dynamic ('Chemistry(Blood)')
    so they fall through to the source's default group."""
    if subgroup in SUBGROUP_TO_GROUP:
        return SUBGROUP_TO_GROUP[subgroup]
    return SOURCES[source].default_group


# --------------------------------------------------- rule-based mapping -----

# procedureevents_mv labels are a small closed vocabulary, so they are mapped
# by an explicit curated list rather than by the LLM. Anything not listed here
# falls through to LLM classification.
PROCEDURE_RULES = {
    "Access & Lines": [
        "Triple Introducer", "Cordis/Introducer", "Multi Lumen", "Tunneled (Hickman) Line",
        "Dialysis Catheter", "Pheresis Catheter", "AVA Line", "PA Catheter", "CCO PAC",
        "Presep Catheter", "Trauma line", "Impella Line", "IABP line",
        "Tandem Heart Access Line", "Tandem Heart Return Line",
        "Temporary Pacemaker Wires Inserted", "Temporary Pacemaker Wires Discontinued",
        "14 Gauge", "16 Gauge", "18 Gauge", "20 Gauge", "22 Gauge",
        "Midline", "PICC Line", "RIC", "Sheath", "Sheath Insertion", "Arterial Line",
        "Unplanned Line/Catheter Removal (Patient Initiated)",
        "Unplanned Line/Catheter Removal (Non-Patient initated)",
    ],
    "Airway & Ventilation": [
        "Intubation", "Extubation",
        "Unplanned Extubation (patient-initiated)",
        "Unplanned Extubation (non-patient initiated)",
        "Open Tracheostomy", "Percutaneous Tracheostomy", "Bronchoscopy",
        "Invasive Ventilation", "Non-invasive Ventilation",
    ],
    "Renal & Dialysis": [
        "Dialysis - CRRT", "Dialysis - CVVHD", "Dialysis - CVVHDF", "Dialysis - SCUF",
        "Peritoneal Dialysis", "Hemodialysis", "CRRT Filter Change",
    ],
    "Cardiac": [
        "EKG", "Cardioversion/Defibrillation", "Cardiac Cath",
        "Transthoracic Echo", "Trans Esophageal Echo", "Pericardiocentesis",
        "Chest Pain", "Cardiac Arrest", "Chest Opened",
    ],
    "Neuro & ICP": [
        "ICP Catheter", "ICP Bolt Inserted", "Intraventricular Drain Inserted",
        "Ventricular Drain", "Lumbar Puncture", "EEG", "Brain Death",
        "C-Spine Clearance", "TLS Clearance", "Pericardial Drain Removed",
    ],
}


# ----------------------------------------------------------- source spec ----

class Source:
    """One MIMIC-III source table's item catalog and how to label it.

    dictionary   how to enumerate the items (see catalog.py)
    event_table  MIMIC csv scanned for TOP_VALUES
    value_cols   column(s) of `event_table` sampled into TOP_VALUES
    value_dtype  'str' | 'numeric' -- pinned rather than inferred, because
                 pandas infers dtypes per chunk and CHARTEVENTS.VALUE would
                 otherwise come back as int in one chunk and str in the next
    strategy     'metadata' | 'rules' | 'llm'
    """

    def __init__(self, name, linksto, dictionary, event_table, join_col, value_cols,
                 strategy, subgroups, default_group, value_desc, label_col="LABEL",
                 value_dtype="str"):
        self.name = name
        self.linksto = linksto
        self.dictionary = dictionary
        self.event_table = event_table
        self.join_col = join_col
        self.value_cols = value_cols
        self.strategy = strategy
        self.subgroups = subgroups
        self.default_group = default_group
        self.value_desc = value_desc
        self.value_dtype = value_dtype
        self.label_col = label_col


SOURCES = {
    "chartevents": Source(
        name="chartevents", linksto="chartevents", dictionary="D_ITEMS",
        event_table="CHARTEVENTS.csv", join_col="ITEMID", value_cols=["VALUE"],
        strategy="llm", subgroups=CHART_SUBGROUPS, default_group=GROUP_ADMIN,
        value_desc="the most frequently recorded values",
    ),
    "outputevents": Source(
        name="outputevents", linksto="outputevents", dictionary="D_ITEMS",
        event_table="OUTPUTEVENTS.csv", join_col="ITEMID", value_cols=["VALUE"],
        strategy="llm", subgroups=OUTPUT_SUBGROUPS, default_group=GROUP_OUTPUT,
        value_desc="the most frequently recorded amounts", value_dtype="numeric",
    ),
    "inputevents_cv": Source(
        name="inputevents_cv", linksto="inputevents_cv", dictionary="D_ITEMS",
        event_table="INPUTEVENTS_CV.csv", join_col="ITEMID",
        value_cols=["AMOUNT", "RATE"],
        strategy="llm", subgroups=INPUT_SUBGROUPS, default_group=GROUP_MEDS,
        value_desc="the most frequently recorded amounts and rates", value_dtype="numeric",
    ),
    "inputevents_mv": Source(
        name="inputevents_mv", linksto="inputevents_mv", dictionary="D_ITEMS",
        event_table="INPUTEVENTS_MV.csv", join_col="ITEMID",
        value_cols=["AMOUNT", "RATE"],
        strategy="llm", subgroups=MED_SUBGROUPS, default_group=GROUP_MEDS,
        value_desc="the most frequently recorded amounts and rates", value_dtype="numeric",
    ),
    "procedureevents_mv": Source(
        name="procedureevents_mv", linksto="procedureevents_mv", dictionary="D_ITEMS",
        event_table="PROCEDUREEVENTS_MV.csv", join_col="ITEMID",
        value_cols=["LOCATION"],
        strategy="rules", subgroups=PROC_SUBGROUPS, default_group=GROUP_PROC,
        value_desc="the most frequent procedure locations",
    ),
    "labevents": Source(
        name="labevents", linksto="Labevents", dictionary="D_LABITEMS",
        event_table="LABEVENTS.csv", join_col="ITEMID", value_cols=["VALUE"],
        strategy="metadata", subgroups=[], default_group=GROUP_LAB,
        value_desc="the most frequently recorded results",
    ),
    "microbiologyevents": Source(
        name="microbiologyevents", linksto="microbiologyevents", dictionary="D_ITEMS",
        event_table="MICROBIOLOGYEVENTS.csv", join_col="SPEC_ITEMID",
        value_cols=["VALUE"],
        strategy="metadata", subgroups=MICRO_SUBGROUPS, default_group=GROUP_MICRO,
        value_desc="the most frequently recorded values",
    ),
    "prescriptions": Source(
        name="prescriptions", linksto="Prescriptions", dictionary="PRESCRIPTIONS",
        event_table="PRESCRIPTIONS.csv", join_col="DRUG", value_cols=["DOSE_VAL_RX"],
        strategy="llm", subgroups=MED_SUBGROUPS, default_group=GROUP_MEDS,
        value_desc="commonly prescribed dose values", label_col="DRUG",
    ),
}

# Prescriptions have no MIMIC ITEMID; synthetic ids are assigned from this base
# so the merged catalog keeps ITEMID unique across source tables.
PRESCRIPTION_ITEMID_BASE = 1_000_000_000
