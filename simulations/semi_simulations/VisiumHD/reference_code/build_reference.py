"""
Builds the single-cell references of the Visium HD colon cancer samples.

Assembles the Chromium Flex atlas from the cache that ``fetch_flex.py`` fills, then reshapes it
onto each sample's annotation vocabulary as ``mapping.yaml`` prescribes. Per sample it writes four
references into ``<sample dir>/references``:

- ``reference_lv0/1/2.h5ad`` — ``obs["cell_type"]`` is exactly the vocabulary of that level, so
  each lines up with the proportions file it is named after;
- ``reference_full.h5ad`` — the finest vocabulary, a strict refinement of all three levels. This
  is the one ``../deconv_code`` deconvolves: one fit per sample, whose proportion columns are then
  summed into each level.

``mapping.csv``, written beside them, is what makes that summing a groupby: one row per fine cell
type, giving its label at each level.

Nothing is written until every check has passed, and the two derived labels (``Paneth cell``,
``Tumor Epithelial Inflamed``) are reported before writing unless ``--write`` is given.

Usage: python build_reference.py [--write] [--samples P1 P2 ...] [--pool-tumour] [--cache DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

sys.dont_write_bytecode = True
from fetch_flex import DEFAULT_CACHE, METADATA_NAME, libraries  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/cluster/CBIO/data1/lgortana")
SAMPLE_DIR = "Visium_HD_Human_Colon_Cancer_{sample}"
LEVELS = ["lv0", "lv1", "lv2"]
DROP = "drop"
SOURCE = {
    "paper": "de Oliveira et al., Nat Genet 57:1512-1523 (2025), doi:10.1038/s41588-025-02193-3",
    "counts": "GEO GSE280318, Chromium Fixed RNA Profiling (Flex), 8 FFPE blocks, 32 libraries",
    "labels": "github.com/10XGenomics/HumanColonCancer_VisiumHD, MetaData/SingleCell_MetaData.csv.gz",
}


# ----------------------------------------------------------------------------- inputs and checks


def sample_levels(sample: str) -> dict:
    """The sample's own level contract, as its pseudo-Visium build wrote it."""

    path = DATA_ROOT / SAMPLE_DIR.format(sample=sample) / "pseudovisium" / "sim" / "levels.json"

    return json.loads(path.read_text())


def check_mapping(mapping: dict, samples: list[str]) -> None:
    """
    Checks ``mapping.yaml`` against the samples' level contracts before any data is read.

    Three ways a mapping can be wrong, all cheap to catch here and expensive to notice later: a
    label that no level defines, a level whose vocabulary is not fully covered, and a group whose
    labels disagree with the sample's own parent/child hierarchy.
    """

    groups = set(mapping["groups"])
    for sample in samples:
        contract = sample_levels(sample)
        by_level = {f"lv{entry['level']}": entry for entry in contract["levels"]}
        table = mapping["samples"][sample]

        if set(table) != groups:
            raise SystemExit(f"{sample}: mapping covers {sorted(set(table))}, expected {sorted(groups)}")

        derived = {name: spec["levels"] for name, spec in mapping["derived"].items() if spec["sample"] == sample}
        for level, entry in by_level.items():
            wanted = set(entry["cell_types"])
            used = {row[level] for row in table.values()} | {row[level] for row in derived.values()}
            used.discard(DROP)
            if used != wanted:
                missing, extra = sorted(wanted - used), sorted(used - wanted)
                raise SystemExit(f"{sample} {level}: missing {missing}, unexpected {extra}")

        for name, row in {**table, **derived}.items():
            for child, parent in (("lv2", "lv1"), ("lv1", "lv0")):
                if row[child] == DROP or row[parent] == DROP:
                    continue
                expected = by_level[child]["parent"][row[child]]
                if expected != row[parent]:
                    raise SystemExit(
                        f"{sample} {name}: {child} {row[child]!r} sits under {expected!r}, "
                        f"but the mapping gives {parent} {row[parent]!r}"
                    )

        shape = "/".join(str(entry["n_types"]) for entry in contract["levels"])
        print(f"  {sample}: mapping agrees with levels.json ({shape} types)")


# -------------------------------------------------------------------------------- the Flex atlas


def read_matrix(path: Path) -> tuple[sp.csr_matrix, np.ndarray, pd.DataFrame]:
    """One Cell Ranger matrix as cells x genes, with its barcodes and features."""

    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        n_genes, n_cells = group["shape"][:]
        # Stored genes x cells in CSC, which is the same buffers as cells x genes in CSR.
        matrix = sp.csr_matrix(
            (group["data"][:].astype(np.float32), group["indices"][:], group["indptr"][:]),
            shape=(n_cells, n_genes),
        )
        barcodes = np.array([b.decode() for b in group["barcodes"][:]])
        features = pd.DataFrame(
            {
                "gene_ids": [g.decode() for g in group["features/id"][:]],
                "symbol": [g.decode() for g in group["features/name"][:]],
                "feature_types": [g.decode() for g in group["features/feature_type"][:]],
                "genome": [g.decode() for g in group["features/genome"][:]],
            }
        )

    return matrix, barcodes, features


def load_atlas(cache: Path, mapping: dict) -> ad.AnnData:
    """
    The Flex atlas: 32 libraries joined to their labels, QC-passing cells only.

    The labels carry ``cellranger aggr`` suffixes ``-1`` .. ``-32`` while every library's own
    barcodes end in ``-1``, so the join needs to know which suffix a library was given. That
    comes from the aggregation order recorded in ``fetch_flex.libraries``, and because a wrong
    guess would join silently against the wrong cells, each library's cell count is checked
    against the number of labels bearing its suffix before anything is joined.
    """

    labels = pd.read_csv(cache / METADATA_NAME)
    labels["suffix"] = labels.Barcode.str.rsplit("-", n=1).str[-1].astype(int)
    labels["core"] = labels.Barcode.str.rsplit("-", n=1).str[0]
    expected = labels.groupby("suffix").size()

    features, blocks, frames = None, [], []
    for entry in libraries():
        matrix, barcodes, found = read_matrix(cache / entry["name"])

        if features is None:
            features = found
        elif not features.equals(found):
            raise SystemExit(f"{entry['name']}: feature table differs from the first library")

        if matrix.shape[0] != expected.get(entry["suffix"], -1):
            raise SystemExit(
                f"{entry['name']}: holds {matrix.shape[0]} cells but suffix {entry['suffix']} "
                f"labels {expected.get(entry['suffix'], 0)} — the aggregation order is wrong"
            )

        cores = pd.Series(barcodes).str.rsplit("-", n=1).str[0]
        frame = pd.DataFrame({"core": cores, "suffix": entry["suffix"], "library": entry["library"]})
        frame["Barcode"] = frame.core + "-" + str(entry["suffix"])
        blocks.append(matrix)
        frames.append(frame)

    obs = pd.concat(frames, ignore_index=True)
    joined = obs.merge(labels, on=["Barcode", "core", "suffix"], how="left", validate="one_to_one")
    if joined.Patient.isna().any():
        raise SystemExit(f"{int(joined.Patient.isna().sum())} barcodes found no label")

    matrix = sp.vstack(blocks, format="csr")
    del blocks

    # Three symbols carry two Ensembl ids each in this probe set. The sample objects keep the
    # first of each pair, so the reference does too and stays indexed the same way.
    keep_gene = ~features.symbol.duplicated(keep="first")
    if not keep_gene.all():
        dropped = features.loc[~keep_gene, ["symbol", "gene_ids"]]
        print(
            f"  dropping {len(dropped)} duplicate symbols (keeping the first id of each): "
            f"{', '.join(dropped.symbol)}"
        )
    matrix = matrix[:, keep_gene.to_numpy()]
    features = features.loc[keep_gene].reset_index(drop=True)

    keep_cell = (joined.QCFilter == "Keep").to_numpy()
    print(f"  {len(joined):,} cells joined, {keep_cell.sum():,} pass QC, {matrix.shape[1]:,} genes")

    var = features.set_index("symbol")
    var.index.name = None
    group_of = {kind: group for group, kinds in mapping["groups"].items() for kind in kinds}
    unmapped = set(joined.Level2[keep_cell]) - set(group_of)
    if unmapped:
        raise SystemExit(f"mapping.yaml has no group for {sorted(unmapped)}")

    atlas = ad.AnnData(
        X=matrix[keep_cell],
        obs=pd.DataFrame(
            {
                "patient": pd.Categorical(joined.Patient[keep_cell]),
                "probe_barcode": pd.Categorical(joined.BC[keep_cell]),
                "library": pd.Categorical(joined.library[keep_cell]),
                "Level1": pd.Categorical(joined.Level1[keep_cell]),
                "Level2": pd.Categorical(joined.Level2[keep_cell]),
                "group": pd.Categorical(joined.Level2[keep_cell].map(group_of)),
            },
            index=pd.Index(joined.Barcode[keep_cell], name=None),
        ),
        var=var,
    )
    atlas.obsm["X_umap"] = joined.loc[keep_cell, ["UMAP1", "UMAP2"]].to_numpy(dtype=np.float32)

    return atlas


# ------------------------------------------------------------------------------ derived labels


def marker_score(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    """Mean log1p counts-per-10k over ``genes``, for the genes the probe set actually carries."""

    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise SystemExit(f"none of {genes} is in the probe set")

    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    block = adata[:, present].X.toarray().astype(np.float64)

    return np.log1p(block / totals[:, None] * 1e4).mean(axis=1)


def carve(atlas: ad.AnnData, name: str, spec: dict, report: bool) -> np.ndarray:
    """
    The cells of ``atlas`` to relabel as ``name``: the top scorers among its parent population.

    The atlas's vocabulary has no such type, so it is carved out by the marker evidence that
    defines it, taking the share of the parent population the sample's own annotation found.
    """

    parent = atlas.obs.group.isin(spec["from_groups"]).to_numpy()
    scores = np.full(atlas.n_obs, -np.inf)
    scores[parent] = marker_score(atlas[parent], spec["markers"])

    eligible = parent.copy()
    if required := spec.get("required"):
        have = [g for g in required if g in atlas.var_names]
        counts = np.asarray(atlas[:, have].X.sum(axis=1)).ravel()
        eligible &= counts > 0

    # Rank within depth strata rather than globally. Detecting a rare transcript takes depth, so
    # the top scorers overall are simply the deepest cells — carving that way gave a Paneth
    # population 1.7x deeper than the goblet cells it came from, which would have handed the
    # deconvolution a profile that differs by library size as much as by biology. Taking the same
    # share out of each decile of the parent's library sizes leaves the markers doing the work.
    wanted = int(round(parent.sum() * spec["fraction"]))
    totals = np.asarray(atlas.X.sum(axis=1)).ravel()
    index = np.flatnonzero(parent)
    strata = pd.qcut(totals[index], q=10, labels=False, duplicates="drop")

    mask = np.zeros(atlas.n_obs, dtype=bool)
    for stratum in np.unique(strata):
        in_stratum = index[strata == stratum]
        quota = int(round(len(in_stratum) * spec["fraction"]))
        candidates = in_stratum[eligible[in_stratum]]
        mask[candidates[np.argsort(-scores[candidates])][:quota]] = True

    if report:
        quantiles = np.percentile(scores[parent], [50, 75, 90, 95, 99])
        print(f"\n  {name}")
        print(f"    parent population ({'+'.join(spec['from_groups'])}): {parent.sum():,} cells")
        if required:
            print(f"    with {'/'.join(required)} counts > 0: {eligible.sum():,} cells")
        print(f"    score percentiles over parent (50/75/90/95/99): " f"{', '.join(f'{q:.3f}' for q in quantiles)}")
        if mask.any():
            print(
                f"    asked for {spec['fraction']:.2%} = {wanted:,}, carved {mask.sum():,} "
                f"(score >= {scores[mask].min():.3f})"
            )
            print(
                f"    median library size: {np.median(totals[mask]):,.0f} carved vs "
                f"{np.median(totals[parent]):,.0f} parent "
                f"({np.median(totals[mask]) / np.median(totals[parent]):.2f}x)"
            )
            taken = atlas.obs.Level2[mask].value_counts()
            print(f"    taken from: {', '.join(f'{k} {v:,}' for k, v in taken.items() if v)}")
            where = atlas.obs.patient[mask].value_counts()
            print(f"    by block:   {', '.join(f'{k} {v:,}' for k, v in where.items() if v)}")
        else:
            print(f"    asked for {spec['fraction']:.2%} = {wanted:,}, carved NOTHING")
        if mask.sum() < wanted:
            print(
                f"    NOTE only {mask.sum():,} of {wanted:,} available — "
                f"the marker requirement, not the fraction, is binding"
            )

    return mask


# -------------------------------------------------------------------------------- shape and write


def for_sample(
    atlas: ad.AnnData, sample: str, mapping: dict, pool_tumour: bool, report: bool
) -> tuple[ad.AnnData, pd.DataFrame]:
    """
    The sample's reference at the finest vocabulary, plus its mapping table.

    ``cell_type`` refines every level: it is the atlas's ``Level2`` except where a derived label
    replaces it, so proportions over it can be summed to any of the three levels.
    """

    keep = np.ones(atlas.n_obs, dtype=bool)
    if not pool_tumour:
        patient = mapping["cells"]["patient_of"][sample]
        private = atlas.obs.group.isin(mapping["cells"]["patient_specific_groups"]).to_numpy()
        foreign = private & (atlas.obs.patient != patient).to_numpy()
        keep &= ~foreign
        if report:
            print(
                f"\n  {sample}: dropping {foreign.sum():,} cells of the patient-specific groups "
                f"from other blocks, keeping {(private & ~foreign).sum():,} from {patient}"
            )

    local = atlas[keep].copy()
    cell_type = local.obs.Level2.astype(str).to_numpy()

    table = dict(mapping["samples"][sample])
    derived_summary = {}
    for name, spec in mapping["derived"].items():
        if spec["sample"] != sample:
            continue
        mask = carve(local, name, spec, report)
        cell_type[mask] = name
        table[name] = spec["levels"]
        local.obs[f"derived_{name.replace(' ', '_')}"] = mask
        parent = int(local.obs.group.isin(spec["from_groups"]).sum())
        asked = int(round(parent * spec["fraction"]))
        derived_summary[name] = {
            "carved": int(mask.sum()),
            "asked": asked,
            "parent_population": parent,
            "markers": spec["markers"],
            "required": spec.get("required", []),
            # When the marker requirement bites before the fraction does, the reference simply
            # does not hold as much of this type as the sample's annotation found. Saying so here
            # keeps the shortfall attached to the data instead of only to a build log.
            "short": int(mask.sum()) < asked,
        }
    local.uns["derived"] = derived_summary

    local.obs["cell_type"] = pd.Categorical(cell_type)

    # One row per fine cell type. Derived types keep their own labels; everything else inherits
    # its group's, so the table is the whole story of how a level was reached.
    group_of = {kind: group for group, kinds in mapping["groups"].items() for kind in kinds}
    rows = []
    for kind in sorted(set(cell_type)):
        group = kind if kind in table else group_of[kind]
        rows.append({"cell_type": kind, "group": group, **{lv: table[group][lv] for lv in LEVELS}})
    frame = pd.DataFrame(rows)

    for level in LEVELS:
        local.obs[level] = pd.Categorical(local.obs.cell_type.astype(str).map(frame.set_index("cell_type")[level]))

    return local, frame


def write(adata: ad.AnnData, path: Path, mapping_table: pd.DataFrame, extra: dict) -> None:
    """Writes one reference, with its mapping and provenance travelling inside it."""

    adata.uns["reference"] = {
        "source": SOURCE,
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        # Column-wise, not records: HDF5 has no way to store a list of dicts, and a column of
        # strings is what both anndata and a later groupby-sum want anyway.
        "mapping": {name: mapping_table[name].astype(str).tolist() for name in mapping_table},
        "derived": adata.uns.get("derived", {}),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, compression="gzip")
    size = path.stat().st_size / 1e9
    print(f"    {path.name:<22} {adata.n_obs:>7,} cells x {adata.n_vars:,} genes  {size:.2f} GB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--samples", nargs="+", default=["P1", "P2", "P5"])
    parser.add_argument("--write", action="store_true", help="write the files instead of reporting only")
    parser.add_argument("--pool-tumour", action="store_true", help="keep all blocks' tumour cells")
    arguments = parser.parse_args()

    mapping = yaml.safe_load((HERE / "mapping.yaml").read_text())

    print("checking mapping.yaml against the samples' levels.json")
    check_mapping(mapping, arguments.samples)

    print("\nloading the Flex atlas")
    atlas = load_atlas(arguments.cache, mapping)

    for sample in arguments.samples:
        local, table = for_sample(atlas, sample, mapping, arguments.pool_tumour, report=True)
        contract = sample_levels(sample)
        by_level = {f"lv{entry['level']}": entry for entry in contract["levels"]}

        per_level = {}
        for level in LEVELS:
            keep = (local.obs[level] != DROP).to_numpy()
            subset = local[keep].copy()
            subset.obs["cell_type"] = pd.Categorical(
                subset.obs[level].astype(str), categories=by_level[level]["cell_types"]
            )
            found = set(subset.obs.cell_type.dropna().unique())
            wanted = set(by_level[level]["cell_types"])
            if found != wanted:
                raise SystemExit(f"{sample} {level}: produced {sorted(found)}, expected {sorted(wanted)}")
            per_level[level] = subset

        print(f"\n  {sample} vocabularies")
        for level in LEVELS:
            counts = per_level[level].obs.cell_type.value_counts()
            print(
                f"    {level} ({len(counts)} types, {per_level[level].n_obs:,} cells): "
                f"{', '.join(f'{k} {v:,}' for k, v in counts.items())}"
            )
        dropped = local.n_obs - per_level["lv2"].n_obs
        print(
            f"    lv2 leaves out {dropped:,} of {local.n_obs:,} cells "
            f"({', '.join(sorted(set(local.obs.Level2[(local.obs.lv2 == DROP).to_numpy()].astype(str))))})"
        )

        if not arguments.write:
            continue

        out = DATA_ROOT / SAMPLE_DIR.format(sample=sample) / "references"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n  writing {out}")
        table.to_csv(out / "mapping.csv", index=False)
        (out / "levels.json").write_text(json.dumps(contract, indent=2))
        write(
            local,
            out / "reference_full.h5ad",
            table,
            {
                "sample": sample,
                "vocabulary": "fine (Level2 refined by derived labels)",
                "note": "deconvolve here and sum proportion columns with mapping.csv",
            },
        )
        for level in LEVELS:
            write(
                per_level[level],
                out / f"reference_{level}.h5ad",
                table,
                {
                    "sample": sample,
                    "vocabulary": level,
                    "annot_column": contract["files"][f"proportions_{level}.csv"],
                    "proportions": f"pseudovisium/sim/proportions_{level}.csv",
                },
            )

    if not arguments.write:
        print("\nreport only — nothing written. Pass --write to write the references.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
