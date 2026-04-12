"""Extract element-level embeddings from MLIPs by averaging over MP-20.

This script should be run in an environment with MLIP dependencies installed
(torch, mace-torch, sevenn, orb-models). It is NOT a dependency of
ElementEmbeddings itself — it produces pre-computed CSV files that get
shipped with the package.

Requirements:
    pip install torch ase mace-torch sevenn orb-models

Usage:
    python scripts/extract_mlip_element_embeddings.py --model mace_mp0
    python scripts/extract_mlip_element_embeddings.py --model sevennet
    python scripts/extract_mlip_element_embeddings.py --model orb_v3
    python scripts/extract_mlip_element_embeddings.py --model all

Output:
    src/elementembeddings/data/element_representations/<model_name>.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import chemical_symbols

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "elementembeddings"
    / "data"
    / "element_representations"
)

# ---------------------------------------------------------------------------
# MP-20 dataset loading
# ---------------------------------------------------------------------------

def load_mp20_structures(mp20_path: str | Path | None = None) -> list[Atoms]:
    """Load structures from the MP-20 dataset.

    If mp20_path is not provided, attempts to download from the
    Materials Project or load from a local cache.
    """
    if mp20_path and Path(mp20_path).exists():
        path = Path(mp20_path)
        if path.suffix == ".xyz" or path.suffix == ".extxyz":
            from ase.io import read
            return read(str(path), index=":")
        elif path.suffix == ".json":
            from ase.io import read
            with open(path) as f:
                data = json.load(f)
            structures = []
            for entry in data:
                if "atoms" in entry:
                    structures.append(read(entry["atoms"]))
                else:
                    # Assume pymatgen Structure dict format
                    from pymatgen.core import Structure
                    struct = Structure.from_dict(entry)
                    structures.append(struct.to_ase_atoms())
            return structures

    # Try loading from matbench/mp_20 via pymatgen
    try:
        from mp_api.client import MPRester
        print("Fetching MP-20 structures from Materials Project API...")
        print("(Set MP_API_KEY environment variable if needed)")
        with MPRester() as mpr:
            docs = mpr.materials.summary.search(
                num_elements=(1, 5),
                num_sites=(1, 20),
                fields=["structure", "material_id"],
            )
        structures = []
        for doc in docs:
            structures.append(doc.structure.to_ase_atoms())
        print(f"Loaded {len(structures)} structures")
        return structures
    except Exception as e:
        raise RuntimeError(
            f"Could not load MP-20 structures: {e}\n"
            "Please provide a path to an extxyz or JSON file with --mp20_path"
        ) from e


# ---------------------------------------------------------------------------
# MACE extraction
# ---------------------------------------------------------------------------

def extract_mace(structures: list[Atoms], model_name: str = "medium") -> dict[int, list[np.ndarray]]:
    """Extract atom-level embeddings from MACE-MP-0."""
    from mace.calculators import mace_mp

    calc = mace_mp(model=model_name, default_dtype="float64")
    element_embeddings = defaultdict(list)

    for i, atoms in enumerate(structures):
        if i % 500 == 0:
            print(f"  MACE: {i}/{len(structures)}")
        try:
            atoms.calc = calc
            descriptors = calc.get_descriptors(atoms, invariants_only=True)
            for j, z in enumerate(atoms.get_atomic_numbers()):
                element_embeddings[z].append(descriptors[j])
        except Exception as e:
            warnings.warn(f"MACE failed on structure {i}: {e}")
            continue

    return element_embeddings


# ---------------------------------------------------------------------------
# SevenNet extraction
# ---------------------------------------------------------------------------

def extract_sevennet(structures: list[Atoms]) -> dict[int, list[np.ndarray]]:
    """Extract atom-level embeddings from SevenNet."""
    import torch
    from sevenn.calculator import SevenNetCalculator

    calc = SevenNetCalculator("7net-omat", device="cpu")
    element_embeddings = defaultdict(list)

    # Register hook to capture embeddings before readout
    captured = {}

    def hook_fn(module, input, output):
        captured["embeddings"] = input[0].detach().cpu().numpy()

    # Find the readout layer and register hook
    model = calc.model
    # SevenNet readout is typically the last linear layer
    for name, module in model.named_modules():
        if "readout" in name.lower() or "output" in name.lower():
            handle = module.register_forward_hook(hook_fn)
            break
    else:
        # Fallback: use the model's forward pass and intercept
        warnings.warn("Could not find readout layer, using last layer")
        modules = list(model.modules())
        handle = modules[-1].register_forward_hook(hook_fn)

    for i, atoms in enumerate(structures):
        if i % 500 == 0:
            print(f"  SevenNet: {i}/{len(structures)}")
        try:
            atoms.calc = calc
            atoms.get_potential_energy()
            if "embeddings" in captured:
                embs = captured["embeddings"]
                for j, z in enumerate(atoms.get_atomic_numbers()):
                    if j < len(embs):
                        element_embeddings[z].append(embs[j])
                captured.clear()
        except Exception as e:
            warnings.warn(f"SevenNet failed on structure {i}: {e}")
            continue

    handle.remove()
    return element_embeddings


# ---------------------------------------------------------------------------
# ORB extraction
# ---------------------------------------------------------------------------

def extract_orb(structures: list[Atoms]) -> dict[int, list[np.ndarray]]:
    """Extract atom-level embeddings from ORB-v3."""
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    orbff = pretrained.orb_v3(device="cpu")
    calc = ORBCalculator(orbff, device="cpu")
    element_embeddings = defaultdict(list)

    # Register hook to capture node features
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            captured["embeddings"] = output.detach().cpu().numpy()
        elif isinstance(output, dict) and "node_features" in output:
            captured["embeddings"] = output["node_features"].detach().cpu().numpy()

    # Find the GNS backbone output
    for name, module in orbff.named_modules():
        if "gns" in name.lower() or "backbone" in name.lower():
            handle = module.register_forward_hook(hook_fn)
            break
    else:
        modules = list(orbff.modules())
        handle = modules[-2].register_forward_hook(hook_fn)

    for i, atoms in enumerate(structures):
        if i % 500 == 0:
            print(f"  ORB: {i}/{len(structures)}")
        try:
            atoms.calc = calc
            atoms.get_potential_energy()
            if "embeddings" in captured:
                embs = captured["embeddings"]
                for j, z in enumerate(atoms.get_atomic_numbers()):
                    if j < len(embs):
                        element_embeddings[z].append(embs[j])
                captured.clear()
        except Exception as e:
            warnings.warn(f"ORB failed on structure {i}: {e}")
            continue

    handle.remove()
    return element_embeddings


# ---------------------------------------------------------------------------
# Aggregation and export
# ---------------------------------------------------------------------------

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
    print(f"Saved {out_path} ({len(df)} elements, {len(df.columns) - 1} dimensions)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXTRACTORS = {
    "mace_mp0": extract_mace,
    "sevennet": extract_sevennet,
    "orb_v3": extract_orb,
}


def main():
    parser = argparse.ArgumentParser(description="Extract MLIP element embeddings")
    parser.add_argument(
        "--model",
        choices=[*EXTRACTORS.keys(), "all"],
        default="all",
        help="Which MLIP to extract from",
    )
    parser.add_argument(
        "--mp20_path",
        type=str,
        default=None,
        help="Path to MP-20 structures (extxyz or JSON)",
    )
    args = parser.parse_args()

    structures = load_mp20_structures(args.mp20_path)
    print(f"Loaded {len(structures)} structures")

    models = EXTRACTORS.keys() if args.model == "all" else [args.model]

    for model_name in models:
        print(f"\nExtracting {model_name}...")
        extractor = EXTRACTORS[model_name]
        element_embeddings = extractor(structures)
        print(f"  Got embeddings for {len(element_embeddings)} elements")
        aggregate_and_save(element_embeddings, model_name)


if __name__ == "__main__":
    main()
