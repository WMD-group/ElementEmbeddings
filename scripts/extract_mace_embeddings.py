"""Extract element-level embeddings from MACE-MP-0 by averaging over MP-20.

Reads CIF strings from the MP-20 train.csv, runs MACE-MP-0, extracts
atom-level descriptors, and mean-pools per element.
"""

from __future__ import annotations

import io
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import chemical_symbols
from ase.io import read as ase_read

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "elementembeddings"
    / "data"
    / "element_representations"
)

MP20_PATH = Path("/Users/aron/Github/PlatonicRep/mp-20/train.csv")

# Use a subset — 5000 structures gives good element coverage
N_STRUCTURES = 5000


def load_structures_from_mp20(path: Path, n: int) -> list:
    """Load ASE Atoms from MP-20 CSV (CIF strings embedded in CSV)."""
    print(f"Loading {n} structures from {path}...")
    df = pd.read_csv(path, nrows=n)

    structures = []
    for i, row in df.iterrows():
        try:
            cif_str = row["cif"]
            atoms = ase_read(io.StringIO(cif_str), format="cif")
            structures.append(atoms)
        except Exception:
            continue

    print(f"  Successfully parsed {len(structures)} structures")
    return structures


def extract_mace(structures: list) -> dict[int, list[np.ndarray]]:
    """Extract atom-level descriptors from MACE-MP-0."""
    from mace.calculators import mace_mp

    print("Loading MACE-MP-0 model...")
    calc = mace_mp(model="medium", default_dtype="float64")

    element_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
    n_failed = 0

    for i, atoms in enumerate(structures):
        if i % 200 == 0:
            print(f"  Processing {i}/{len(structures)}...")
        try:
            atoms.calc = calc
            descriptors = calc.get_descriptors(atoms, invariants_only=True)
            for j, z in enumerate(atoms.get_atomic_numbers()):
                element_embeddings[z].append(descriptors[j])
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                warnings.warn(f"Failed on structure {i}: {e}")
            continue

    if n_failed > 0:
        print(f"  {n_failed} structures failed")

    return element_embeddings


def aggregate_and_save(
    element_embeddings: dict[int, list[np.ndarray]],
    model_name: str,
) -> None:
    """Mean-pool per element and save as CSV."""
    rows = []
    for z in sorted(element_embeddings.keys()):
        embs = np.array(element_embeddings[z])
        mean_emb = embs.mean(axis=0)
        symbol = chemical_symbols[z]
        rows.append({"element": symbol, **{str(i): v for i, v in enumerate(mean_emb)}})

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / f"{model_name}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {out_path}")
    print(f"  {len(df)} elements, {len(df.columns) - 1} dimensions")
    print(f"  Elements: {', '.join(df['element'].tolist())}")

    # Print sample counts per element
    counts = {chemical_symbols[z]: len(v) for z, v in sorted(element_embeddings.items())}
    min_el = min(counts, key=counts.get)
    max_el = max(counts, key=counts.get)
    print(f"  Min samples: {min_el} ({counts[min_el]}), Max samples: {max_el} ({counts[max_el]})")


if __name__ == "__main__":
    structures = load_structures_from_mp20(MP20_PATH, N_STRUCTURES)
    element_embeddings = extract_mace(structures)
    print(f"\nGot embeddings for {len(element_embeddings)} elements")
    aggregate_and_save(element_embeddings, "mace_mp0")
