from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ManifestSummary:
    rows: int
    direction_pairs: int
    factorial_tuples: int
    factorial_tuple_variants: int
    carrier_variants: tuple[str, ...]
    split_rows: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def validate_manifest(df: pd.DataFrame) -> None:
    required = {"image", "instruction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    if {"direction_pair_id", "direction_label"}.issubset(df.columns):
        pairs = df[df["direction_pair_id"].astype(str) != ""]
        for pid, group in pairs.groupby("direction_pair_id", dropna=False):
            labels = pd.to_numeric(group["direction_label"], errors="coerce").dropna().astype(int).tolist()
            if sorted(labels) != [0, 1]:
                raise ValueError(f"direction pair {pid!r} must contain exactly one benign and one harmful row")

    factorial = df[df["tuple_id"].astype(str) != ""] if "tuple_id" in df else df.iloc[0:0]
    if not factorial.empty:
        needed = {"tuple_id", "source_level", "edit_level", "variant"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"factorial rows require columns: {sorted(missing)}")
        for (tid, variant), group in factorial.groupby(["tuple_id", "variant"], dropna=False):
            if len(group) != 4:
                raise ValueError(f"tuple={tid}, variant={variant} must contain exactly four factorial rows")
            cells = {(int(s), int(a)) for s, a in zip(group.source_level, group.edit_level)}
            if cells != {(0, 0), (0, 1), (1, 0), (1, 1)}:
                raise ValueError(f"tuple={tid}, variant={variant} does not contain exactly the 2x2 cells")
            if "risk_label" in group:
                expected = (group.source_level.astype(int) & group.edit_level.astype(int)).to_numpy()
                actual = pd.to_numeric(group.risk_label, errors="raise").astype(int).to_numpy()
                if not (expected == actual).all():
                    raise ValueError(
                        f"tuple={tid}, variant={variant}: risk_label must be 1 only for the (S1,A1) cell"
                    )

    # Split-separation claims apply to the monitor train/dev/test groups. The
    # separate direction-calibration pool is intentionally excluded here.
    if {"split", "source_group"}.issubset(df.columns):
        _check_disjoint(df, "source_group")
    if {"split", "normalized_edit_template"}.issubset(df.columns):
        _check_disjoint(df, "normalized_edit_template")


def manifest_summary(df: pd.DataFrame) -> ManifestSummary:
    pair_count = 0
    if "direction_pair_id" in df:
        pair_count = df.loc[df.direction_pair_id.astype(str) != "", "direction_pair_id"].astype(str).nunique()
    tuple_count = 0
    tuple_variant_count = 0
    carrier_variants: tuple[str, ...] = ()
    if "tuple_id" in df:
        fact = df[df.tuple_id.astype(str) != ""]
        tuple_count = fact.tuple_id.astype(str).nunique()
        if not fact.empty and "variant" in fact:
            tuple_variant_count = fact[["tuple_id", "variant"]].astype(str).drop_duplicates().shape[0]
            carrier_variants = tuple(sorted(fact.variant.astype(str).unique()))
    split_rows = {}
    if "split" in df:
        split_rows = {str(k): int(v) for k, v in df.split.astype(str).value_counts().to_dict().items()}
    return ManifestSummary(
        rows=int(len(df)),
        direction_pairs=int(pair_count),
        factorial_tuples=int(tuple_count),
        factorial_tuple_variants=int(tuple_variant_count),
        carrier_variants=carrier_variants,
        split_rows=split_rows,
    )


def _check_disjoint(df: pd.DataFrame, field: str) -> None:
    work = df[["split", field]].copy()
    work["split"] = work["split"].astype(str)
    work[field] = work[field].astype(str)
    work = work[work["split"].isin(["train", "dev", "test"])]
    work = work[work[field] != ""]
    crossing = work.groupby(field)["split"].nunique()
    bad = crossing[crossing > 1]
    if len(bad):
        examples = list(bad.index[:8])
        raise ValueError(f"{field} appears in multiple train/dev/test splits; examples={examples}")
