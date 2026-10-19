"""
Ontology induction -- discover the GROUP / SUBGROUP tree from the item labels
themselves, given nothing but MIMIC-III and an LLM.

Two levels, same loop shape:

    seed      show the first batch of items, ask for an initial category set
    extend    sweep the remaining items batch by batch; each batch may propose
              categories only if several of its items have no home yet
    settle    stop after `patience` consecutive batches propose nothing new
    consolidate  one pass that merges near-duplicate categories

GROUP induction sweeps every item. SUBGROUP induction runs the same loop once
per group, over only the items assigned to that group -- so subgroups are
discovered in the context of their parent, not against the whole database.

Nothing here is seeded from `taxonomy.py`; the only inputs are item labels and
their TOP_VALUES.
"""
import os

from ontology import Ontology, norm_name

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt")

# Batch sizes are deliberately smaller than for assignment: the model has to
# reason about the whole batch at once rather than emit one line per item.
SEED_BATCH = int(os.getenv("INDUCE_SEED_BATCH", "200"))
EXTEND_BATCH = int(os.getenv("INDUCE_EXTEND_BATCH", "150"))
PATIENCE = int(os.getenv("INDUCE_PATIENCE", "3"))

GROUP_TARGET = (int(os.getenv("INDUCE_GROUP_MIN", "8")),
                int(os.getenv("INDUCE_GROUP_MAX", "16")))
SUBGROUP_TARGET = (int(os.getenv("INDUCE_SUBGROUP_MIN", "3")),
                   int(os.getenv("INDUCE_SUBGROUP_MAX", "12")))


def _prompt(name):
    return open(os.path.join(PROMPT_DIR, name), encoding="utf-8").read()


def _render_items(items):
    """items: list of (label, top_values). TOP_VALUES may be blank."""
    lines = []
    for label, values in items:
        values = str(values).strip()
        lines.append(f"{label} : {values}" if values and values not in ("[]", "{}", "nan")
                     else str(label))
    return "\n".join(lines)


def _batches(items, size):
    for start in range(0, len(items), size):
        yield start // size, items[start:start + size]


# --------------------------------------------------------------- parsing ----

def parse_categories(text):
    """Parse '- name | definition[ | evidence]' lines. Returns [(name, definition)].

    The pipe is required. Both prompts mandate it, and requiring it is what
    keeps a conversational preamble ("Here you go:") or a trailing remark from
    being registered as a category -- which silently corrupts the ontology.
    """
    out = []
    for line in (text or "").strip().splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line or line.upper() == "NONE" or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        name = norm_name(parts[0])
        # Guard against the model echoing the output format itself.
        if not name or len(name) > 80 or name.lower().startswith("category name"):
            continue
        out.append((name, parts[1] if len(parts) > 1 else ""))
    return out


def parse_merges(text):
    """Parse 'a => b' lines. Returns [(remove, keep)]."""
    out = []
    for line in (text or "").strip().splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line or line.upper() == "NONE" or "=>" not in line:
            continue
        remove, _, keep = line.partition("=>")
        remove, keep = norm_name(remove), norm_name(keep)
        if remove and keep and remove.lower() != keep.lower():
            out.append((remove, keep))
    return out


# ------------------------------------------------------------- the loop -----

def induce_level(items, client, model, level, scope, target, onto,
                 group=None, seed_batch=SEED_BATCH, extend_batch=EXTEND_BATCH,
                 patience=PATIENCE, consolidate=True, log=print):
    """Run seed -> extend -> settle for one level. Mutates and returns `onto`.

    items   [(label, top_values)], already ordered
    level   'group' or 'subgroup' (used in the prompts)
    scope   one sentence telling the model what population it is organizing
    group   parent group when inducing subgroups, else None
    """
    from models import generate

    def current():
        return onto.as_prompt_list(group=group)

    def add(name, description):
        if group is None:
            return onto.add_group(name, description)
        return onto.add_subgroup(group, name, description)

    def known():
        return onto.group_names() if group is None else onto.subgroup_names(group)

    rounds, proposed_total = 0, 0

    # ---- seed
    if not known():
        head = items[:seed_batch]
        prompt = (_prompt("induce_seed.txt")
                  .replace("<<<LEVEL>>>", level)
                  .replace("<<<SCOPE>>>", scope)
                  .replace("<<<TARGET_MIN>>>", str(target[0]))
                  .replace("<<<TARGET_MAX>>>", str(target[1]))
                  .replace("<<<INPUT>>>", _render_items(head)))
        for name, desc in parse_categories(generate(client, prompt, model=model)):
            add(name, desc)
        rounds += 1
        log(f"    seed ({len(head)} items) -> {len(known())} {level}(s)")
        items = items[seed_batch:]

    if not known():
        log(f"    [WARN] seed produced no {level}s")
        return onto

    # ---- extend until `patience` consecutive quiet batches
    quiet = 0
    template = _prompt("induce_extend.txt")
    for idx, batch in _batches(items, extend_batch):
        if quiet >= patience:
            log(f"    settled after {idx} extend batch(es); "
                f"{len(items) - idx * extend_batch} item(s) not swept")
            break
        prompt = (template
                  .replace("<<<LEVEL>>>", level)
                  .replace("<<<SCOPE>>>", scope)
                  .replace("<<<CATEGORIES>>>", current())
                  .replace("<<<INPUT>>>", _render_items(batch)))
        new = parse_categories(generate(client, prompt, model=model))
        # A proposal that only restates an existing category is not new.
        fresh = [(n, d) for n, d in new
                 if not (onto.has_group(n) if group is None
                         else onto.has_subgroup(group, n))]
        rounds += 1
        if fresh:
            for name, desc in fresh:
                add(name, desc)
            proposed_total += len(fresh)
            quiet = 0
            log(f"    batch {idx + 1}: +{len(fresh)} "
                f"({', '.join(n for n, _ in fresh)}) -> {len(known())} total")
        else:
            quiet += 1
    else:
        log(f"    swept all items; {len(known())} {level}(s)")

    # ---- consolidate. Always run: growing the set batch by batch is exactly
    # what produces near-duplicates, and that happens well below the cap.
    if consolidate and len(known()) > 1:
        prompt = (_prompt("induce_consolidate.txt")
                  .replace("<<<LEVEL>>>", level)
                  .replace("<<<SCOPE>>>", scope)
                  .replace("<<<CATEGORIES>>>", current())
                  .replace("<<<TARGET_MAX>>>", str(target[1])))
        merges = parse_merges(generate(client, prompt, model=model))
        applied = 0
        for remove, keep in merges:
            ok = (onto.merge_groups(remove, keep) if group is None
                  else _merge_subgroup(onto, group, remove, keep))
            applied += bool(ok)
        rounds += 1
        log(f"    consolidated: {applied} merge(s) -> {len(known())} {level}(s)")

    onto.provenance.setdefault("rounds", {})[f"{level}:{group or 'all'}"] = rounds
    onto.provenance.setdefault("proposed", {})[f"{level}:{group or 'all'}"] = proposed_total
    return onto


def _merge_subgroup(onto, group, remove, keep):
    g = onto.resolve_group(group)
    src = onto.resolve_subgroup(g, remove)
    dst = onto.resolve_subgroup(g, keep)
    if not src or not dst or src == dst:
        return False
    del onto.groups[g]["subgroups"][src]
    return True


# -------------------------------------------------------------- drivers -----

def induce_groups(items, client, model, onto=None, log=print, **kwargs):
    """Sweep every item and grow the top-level GROUP set."""
    onto = onto or Ontology()
    scope = ("You are organizing every item recorded for a patient in an ICU database: "
             "vital signs, laboratory results, medications and fluids, intake and output, "
             "procedures, microbiology, and administrative entries.")
    log(f"  inducing groups over {len(items)} item(s)")
    induce_level(items, client, model, "group", scope, GROUP_TARGET, onto,
                 group=None, log=log, **kwargs)
    onto.provenance["n_items_seen"] = len(items)
    return onto


def induce_subgroups(items_by_group, client, model, onto, log=print, **kwargs):
    """For each group, induce its SUBGROUPs from that group's items only."""
    for group in onto.group_names():
        items = items_by_group.get(group, [])
        if not items:
            log(f"  [skip] {group}: no items assigned")
            continue
        log(f"  {group}: {len(items)} item(s)")
        scope = (f"You are organizing the items that were already assigned to the "
                 f"category \"{group}\" ({onto.groups[group].get('description', '')}). "
                 f"Subdivide only within that category.")
        induce_level(items, client, model, "subgroup", scope, SUBGROUP_TARGET, onto,
                     group=group, log=log, **kwargs)
    return onto
