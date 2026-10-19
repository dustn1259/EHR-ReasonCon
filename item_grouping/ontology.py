"""
The induced ontology: a two-level GROUP / SUBGROUP tree discovered from the
item labels themselves rather than written by hand.

Persisted as JSON so it can be inspected, hand-edited, and version-controlled
between runs:

    {
      "groups": {
        "Hemodynamics & Vitals": {
          "description": "Continuously monitored circulatory measurements ...",
          "subgroups": {
            "Blood pressure": "Arterial, non-invasive and central pressures",
            ...
          }
        },
        ...
      },
      "provenance": {"n_items_seen": 17487, "group_rounds": 4, ...}
    }

`taxonomy.py` holds the fixed, published taxonomy used to reproduce the paper.
This module holds a taxonomy induced from scratch. Both feed the same
assignment code -- see `assign.py`.
"""
import json
import os
import re


def norm_name(name):
    """Canonical form for comparing two category names."""
    s = re.sub(r"\s+", " ", str(name).strip())
    s = s.strip(" .-–—:;")
    return s


def _key(name):
    return norm_name(name).lower()


class Ontology:
    def __init__(self, groups=None, provenance=None):
        # {group_name: {"description": str, "subgroups": {name: description}}}
        self.groups = groups or {}
        self.provenance = provenance or {}

    # ------------------------------------------------------------- groups --

    def group_names(self):
        return list(self.groups)

    def has_group(self, name):
        return _key(name) in {_key(g) for g in self.groups}

    def resolve_group(self, name):
        """Return the stored spelling of a group, or None."""
        target = _key(name)
        for g in self.groups:
            if _key(g) == target:
                return g
        return None

    def add_group(self, name, description=""):
        """Add a group. Returns the canonical name; existing groups are kept
        as-is so a later round cannot silently rename an established category."""
        existing = self.resolve_group(name)
        if existing:
            if description and not self.groups[existing].get("description"):
                self.groups[existing]["description"] = description
            return existing
        name = norm_name(name)
        self.groups[name] = {"description": description, "subgroups": {}}
        return name

    def merge_groups(self, source, target):
        """Fold `source` into `target`, moving its subgroups across."""
        src, dst = self.resolve_group(source), self.resolve_group(target)
        if not src or not dst or src == dst:
            return False
        for sub, desc in self.groups[src]["subgroups"].items():
            self.groups[dst]["subgroups"].setdefault(sub, desc)
        del self.groups[src]
        return True

    def rename_group(self, old, new):
        src = self.resolve_group(old)
        if not src:
            return False
        if self.resolve_group(new) and self.resolve_group(new) != src:
            return self.merge_groups(src, new)
        self.groups[norm_name(new)] = self.groups.pop(src)
        return True

    # ---------------------------------------------------------- subgroups --

    def subgroup_names(self, group):
        g = self.resolve_group(group)
        return list(self.groups[g]["subgroups"]) if g else []

    def has_subgroup(self, group, name):
        return _key(name) in {_key(s) for s in self.subgroup_names(group)}

    def resolve_subgroup(self, group, name):
        target = _key(name)
        for s in self.subgroup_names(group):
            if _key(s) == target:
                return s
        return None

    def add_subgroup(self, group, name, description=""):
        g = self.resolve_group(group)
        if not g:
            raise KeyError(f"unknown group {group!r}")
        existing = self.resolve_subgroup(g, name)
        if existing:
            if description and not self.groups[g]["subgroups"][existing]:
                self.groups[g]["subgroups"][existing] = description
            return existing
        name = norm_name(name)
        self.groups[g]["subgroups"][name] = description
        return name

    def group_of_subgroup(self, subgroup):
        """Reverse lookup. Ambiguous names resolve to the first owning group."""
        target = _key(subgroup)
        for g, spec in self.groups.items():
            if any(_key(s) == target for s in spec["subgroups"]):
                return g
        return None

    # ------------------------------------------------------------ summary --

    def as_prompt_list(self, group=None, with_descriptions=True):
        """Render the current categories for inclusion in a prompt."""
        if group is None:
            items = [(g, spec["description"]) for g, spec in self.groups.items()]
        else:
            g = self.resolve_group(group)
            items = list(self.groups[g]["subgroups"].items()) if g else []
        if not items:
            return "(none yet)"
        lines = []
        for i, (name, desc) in enumerate(items, 1):
            lines.append(f"{i}. {name}" + (f" -- {desc}" if with_descriptions and desc else ""))
        return "\n".join(lines)

    def summary(self):
        n_sub = sum(len(s["subgroups"]) for s in self.groups.values())
        return f"{len(self.groups)} group(s), {n_sub} subgroup(s)"

    # --------------------------------------------------------------- i/o ---

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"groups": self.groups, "provenance": self.provenance},
                      fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(groups=data.get("groups", {}), provenance=data.get("provenance", {}))

    @classmethod
    def from_taxonomy(cls, catalog=None):
        """Build an Ontology from the fixed taxonomy in `taxonomy.py`, so the
        published categories can be used by the same assignment code.

        `SUBGROUP_TO_GROUP` only holds the *static* names. Lab and microbiology
        subgroups are derived from MIMIC metadata at build time
        ("Chemistry(Blood)", "Hematology(Urine)", ...) and never appear there,
        so without `catalog` the entire Labs group would collapse to a single
        subgroup. Pass the built catalog to pick those up.
        """
        import taxonomy as tx
        onto = cls()
        for subgroup, group in tx.SUBGROUP_TO_GROUP.items():
            onto.add_group(group)
            onto.add_subgroup(group, subgroup)

        harvested = 0
        if catalog is not None:
            rows = catalog[["GROUP", "SUBGROUP"]].fillna("").astype(str)
            for group, subgroup in rows.drop_duplicates().itertuples(index=False):
                group, subgroup = group.strip(), subgroup.strip()
                if not group or not subgroup:
                    continue
                onto.add_group(group)
                if not onto.has_subgroup(group, subgroup):
                    onto.add_subgroup(group, subgroup)
                    harvested += 1

        onto.provenance = {"source": "taxonomy.py (fixed, published)",
                           "harvested_from_catalog": harvested}
        return onto
