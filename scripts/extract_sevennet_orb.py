"""Extract element-level embeddings from SevenNet and ORB over MP-20."""

from __future__ import annotations

import io
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
N_STRUCTURES = 5000


def load_structures(path: Path, n: int) -> list:
    print(f"Loading {n} structures from {path}...")
    df = pd.read_csv(path, nrows=n)
    structures = []
    for _, row in df.iterrows():
        try:
            atoms = ase_read(io.StringIO(row["cif"]), format="cif")
            structures.append(atoms)
        except Exception:
            continue
    print(f"  Parsed {len(structures)} structures")
    return structures


def aggregate_and_save(element_embeddings: dict, model_name: str) -> None:
    rows = []
    for z in sorted(element_embeddings.keys()):
        embs = np.array(element_embeddings[z])
        mean_emb = embs.mean(axis=0)
        symbol = chemical_symbols[z]
        rows.append({"element": symbol, **{str(i): v for i, v in enumerate(mean_emb)}})
    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / f"{model_name}.csv"
    df.to_csv(out_path, index=False)
    counts = {chemical_symbols[z]: len(v) for z, v in sorted(element_embeddings.items())}
    min_el = min(counts, key=counts.get)
    max_el = max(counts, key=counts.get)
    print(f"Saved {out_path}: {len(df)} elements, {len(df.columns) - 1}D")
    print(f"  Min samples: {min_el} ({counts[min_el]}), Max: {max_el} ({counts[max_el]})")


# ---------------------------------------------------------------------------
# SevenNet
# ---------------------------------------------------------------------------

def extract_sevennet(structures: list) -> dict[int, list[np.ndarray]]:
    from sevenn.calculator import SevenNetCalculator

    print("Loading SevenNet 7net-0...")
    calc = SevenNetCalculator("7net-0", device="cpu")

    captured = {}

    # Hook the last equivariant gate (layer before reduce_input_to_hidden)
    # to capture 128D scalar (l=0) node features
    layer_names = list(calc.model._modules.keys())
    idx = layer_names.index("reduce_input_to_hidden")
    target = layer_names[idx - 1]
    print(f"  Hooking: {target}")

    def hook_fn(module, input, output):
        captured["x"] = output.x.detach().cpu().numpy()
        captured["z"] = output.atomic_numbers.detach().cpu().numpy()

    calc.model._modules[target].register_forward_hook(hook_fn)

    element_embeddings = defaultdict(list)
    n_failed = 0
    for i, atoms in enumerate(structures):
        if i % 200 == 0:
            print(f"  SevenNet: {i}/{len(structures)}")
        try:
            atoms.calc = calc
            atoms.get_potential_energy()
            if "x" in captured:
                for j, z in enumerate(captured["z"]):
                    element_embeddings[int(z)].append(captured["x"][j])
                captured.clear()
        except Exception:
            n_failed += 1
            continue

    if n_failed:
        print(f"  {n_failed} structures failed")
    return element_embeddings


# ---------------------------------------------------------------------------
# ORB
# ---------------------------------------------------------------------------

def extract_orb(structures: list) -> dict[int, list[np.ndarray]]:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    print("Loading ORB-v2...")
    model, adapter = pretrained.orb_v2(device="cpu")

    captured = {}

    # Hook the energy head to capture node features fed into prediction
    def hook_fn(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            x = input[0]
            if isinstance(x, torch.Tensor):
                captured["emb"] = x.detach().cpu().numpy()

    handle = model.heads["energy"].register_forward_hook(hook_fn)

    calc = ORBCalculator(model, adapter, device="cpu")
    element_embeddings = defaultdict(list)
    n_failed = 0

    for i, atoms in enumerate(structures):
        if i % 200 == 0:
            print(f"  ORB: {i}/{len(structures)}")
        try:
            atoms.calc = calc
            atoms.get_potential_energy()
            if "emb" in captured:
                for j, z in enumerate(atoms.get_atomic_numbers()):
                    if j < len(captured["emb"]):
                        element_embeddings[z].append(captured["emb"][j])
                captured.clear()
        except Exception:
            n_failed += 1
            continue

    handle.remove()
    if n_failed:
        print(f"  {n_failed} structures failed")
    return element_embeddings


if __name__ == "__main__":
    import sys

    structures = load_structures(MP20_PATH, N_STRUCTURES)

    model = sys.argv[1] if len(sys.argv) > 1 else "sevennet"

    if model in ("sevennet", "all"):
        print("\n=== SevenNet ===")
        embs = extract_sevennet(structures)
        print(f"Got {len(embs)} elements")
        aggregate_and_save(embs, "sevennet")

    if model in ("orb", "all"):
        print("\n=== ORB ===")
        embs = extract_orb(structures)
        print(f"Got {len(embs)} elements")
        aggregate_and_save(embs, "orb_v2")
