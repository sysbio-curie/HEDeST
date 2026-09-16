"""Build one transformed single-cell reference per (sample, level).

Recipe, identical for every reference in benchmark/results/PanoSpace/deconv/refs:
map each atlas cell type onto the level's vocabulary (celltype_rules), cap at
2,000 cells per class, keep **raw counts** in `.X`, drop `.raw`, and record the
original label in `orig_cell_type` so every mapping stays auditable.

Two details that matter and are easy to get wrong:

* the cap is not cosmetic -- RCTD densifies the whole reference matrix, and the
  full skin atlas would need ~106 GB;
* `.raw` is dropped deliberately. cell2location does
  ``adata.X = adata.raw.X.copy()`` inside a bare ``try``, so a reference that
  keeps a `.raw` is silently deconvolved from it -- and the DISCO atlases ship
  a malformed one whose X has a single row against ~10^5 obs.

    python build_refs.py --sample lung_s3 --level 3 --atlas lung \
        --out <refs dir> --tmp <scratch dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from celltype_rules import map_types, fine_vocabulary
from hierarchy import parent_map

REFS = "/cluster/CBIO/data1/lgortana/STHELAR/references"
BENCH = "/cluster/CBIO/data1/lgortana/STHELAR/bench_data"
CAP, SEED = 2000, 42


def load_atlas(name, tmp):
    """``(cell-type Series, opener(rows) -> AnnData)``.

    DISCO atlases are opened backed so only the sampled rows are materialised.
    The 3CA prostate bundle is an MTX triple (genes x cells, so transposed);
    it is converted once and cached as .npz in `tmp`.
    """
    if name == "prostate":
        d = os.path.join(tmp, "Data_Song2022_Prostate")
        if not os.path.isdir(d):
            raise SystemExit(f"extract references/prostate.tar.gz into {tmp} first")
        cells = pd.read_csv(os.path.join(d, "Cells.csv")).set_index("cell_name")
        genes = pd.read_csv(os.path.join(d, "Genes.txt"), header=None)[0].astype(str)
        cache = os.path.join(tmp, "prostate_X.npz")
        if os.path.exists(cache):
            X = sp.csr_matrix(sp.load_npz(cache))
        else:
            from scipy.io import mmread

            X = sp.csr_matrix(mmread(os.path.join(d, "Exp_data_UMIcounts.mtx")).T)
            sp.save_npz(cache, X)
        ct = cells["cell_type"].astype(str)

        def opener(rows):
            return ad.AnnData(
                X=X[rows].astype(np.float64),
                obs=cells.iloc[rows].copy(),
                var=pd.DataFrame(index=pd.Index(genes, name="gene")),
            )

        return ct, opener

    a = ad.read_h5ad(os.path.join(REFS, f"{name}.h5ad"), backed="r")
    ct = a.obs["cell_type"].astype(str)

    def opener(rows):
        # X/obs/var only. `a[rows].to_memory()` would also subset `.raw`, which
        # these files store malformed, and raises.
        X = a.X[rows]
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        return ad.AnnData(X=X.astype(np.float64), obs=a.obs.iloc[rows].copy(), var=a.var.copy())

    return ct, opener


def vocabulary(sample, level):
    f = f"{BENCH}/{sample}/sim/level{level}/proportions.csv"
    return list(pd.read_csv(f, nrows=0, index_col=0).columns)


def build(sample, level, atlas, out_dir, tmp, granulosa=True, tag="", fine=False, via_level2=False):
    """`fine=True` builds the level-2 vocabulary with its structural classes
    split into Endothelial / Perivascular / Fibroblast -- one reference per
    sample, aggregated afterwards to whichever level is wanted.

    `via_level2=True` maps every atlas type to the **level-2** vocabulary first
    and then rolls the assignment up to `level`, so a type with no level-2 home
    is absent from the reference at every level and a sample's levels are built
    from exactly the same cells. Levels 3 and 4 are finer than 2 and cannot be
    reached this way.
    """
    vocab = fine_vocabulary(vocabulary(sample, 2)) if fine else vocabulary(sample, level)
    ct, opener = load_atlas(atlas, tmp)
    if via_level2 and not fine:
        if level > 2:
            raise SystemExit("--via-level2 cannot reach a level finer than 2")
        assigned_l2, dropped = map_types(sorted(ct.unique()), vocabulary(sample, 2), granulosa_as_epithelial=granulosa)
        up = parent_map(sample, level, 2)
        assigned = {t: up[c] for t, c in assigned_l2.items()}
    else:
        assigned, dropped = map_types(sorted(ct.unique()), vocab, granulosa_as_epithelial=granulosa)
    missing = [c for c in vocab if c not in set(assigned.values())]
    if missing:
        # A level is only attempted when every class has atlas support; this is
        # the guard that enforces it rather than producing a hollow signature.
        raise SystemExit(f"{sample} level{level}: no atlas type maps to {missing}")

    target = ct.map(assigned)
    rng = np.random.default_rng(SEED)
    rows, comp = [], {}
    for cls in vocab:
        idx = np.where((target == cls).to_numpy())[0]
        take = idx if len(idx) <= CAP else rng.choice(idx, CAP, replace=False)
        rows.extend(sorted(take.tolist()))
        comp[cls] = {"n_pool": int(len(idx)), "n_used": int(len(take))}
    rows = np.array(sorted(rows))

    sub = opener(rows)
    sub.obs["orig_cell_type"] = ct.iloc[rows].to_numpy()
    sub.obs["cell_type"] = target.iloc[rows].to_numpy()  # PanoSpace --celltype-key
    sub.obs["ref_level"] = int(level)
    name = f"{sample}_level{level}{tag}"
    os.makedirs(out_dir, exist_ok=True)
    sub.write_h5ad(os.path.join(out_dir, f"{name}.h5ad"))

    meta = {
        "sample": sample,
        "level": level,
        "reference": atlas,
        "fine": fine,
        "granulosa_as_epithelial": granulosa,
        "via_level2": via_level2,
        "vocabulary": vocab,
        "n_cells_atlas": int(len(ct)),
        "n_cells_used": int(len(rows)),
        "per_class": comp,
        "map": {k: {"n": int((ct == k).sum()), "class": v} for k, v in sorted(assigned.items())},
        "dropped": {k: {"n": int((ct == k).sum()), "why": w} for k, w in sorted(dropped.items())},
    }
    with open(os.path.join(out_dir, f"{name}.map.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  {name}: {len(rows):,} cells, {len(vocab)} classes, " f"{len(dropped)} atlas type(s) dropped")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--atlas", required=True, help="breast | lung | skin | ovary | prostate | bone_marrow")
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", required=True)
    p.add_argument("--no-granulosa", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument(
        "--via-level2",
        action="store_true",
        help="map to the level-2 vocabulary and roll up (round 1's " "convention); levels 0-2 only",
    )
    p.add_argument(
        "--fine",
        action="store_true",
        help="build the fine-grained reference (level-2 vocabulary with "
        "Blood_vessel split into Endothelial + Perivascular); one "
        "per sample, aggregated later by aggregate_fine.py",
    )
    a = p.parse_args()
    build(
        a.sample,
        a.level,
        a.atlas,
        a.out,
        a.tmp,
        granulosa=not a.no_granulosa,
        tag=a.tag,
        fine=a.fine,
        via_level2=a.via_level2,
    )
