"""Extract element embeddings from MatSciBERT token embedding layer.

Uses the pre-trained MatSciBERT model from HuggingFace to get a 768D
vector for each element symbol via its token embedding.

Requirements:
    pip install transformers torch
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "elementembeddings"
    / "data"
    / "element_representations"
    / "matscibert.csv"
)

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
]

print("Loading MatSciBERT from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained("m3rg-iitd/matscibert")
model = AutoModel.from_pretrained("m3rg-iitd/matscibert")
model.eval()

embed_weights = model.embeddings.word_embeddings.weight.detach().numpy()

rows = []
skipped = []
for el in ELEMENTS:
    ids = tokenizer.encode(el, add_special_tokens=False)
    if len(ids) == 1:
        vec = embed_weights[ids[0]]
        rows.append({"element": el, **{str(i): v for i, v in enumerate(vec)}})
    else:
        # Multi-token: average the sub-token embeddings
        vecs = embed_weights[ids]
        vec = vecs.mean(axis=0)
        rows.append({"element": el, **{str(i): v for i, v in enumerate(vec)}})
        skipped.append((el, tokenizer.tokenize(el)))

if skipped:
    print(f"Multi-token elements (averaged): {skipped}")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT}: {len(df)} elements, {len(df.columns)-1}D")
