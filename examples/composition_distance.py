"""Illustrative example: distance between compositions using different embeddings."""

from elementembeddings.composition import CompositionalEmbedding

# MACE-MP-0 (learned from atomistic simulations)
zno = CompositionalEmbedding("ZnO", "mace_mp0")
cdo = CompositionalEmbedding("CdO", "mace_mp0")
hgxe = CompositionalEmbedding("HgXe", "mace_mp0")

print("MACE-MP-0")
print(f"  ZnO vs CdO   Euclidean: {zno.distance(cdo, 'euclidean'):.4f}  Cosine: {zno.distance(cdo, 'cosine_distance'):.4f}")
print(f"  ZnO vs HgXe  Euclidean: {zno.distance(hgxe, 'euclidean'):.4f}  Cosine: {zno.distance(hgxe, 'cosine_distance'):.4f}")
# MACE-MP-0
#   ZnO vs CdO   Euclidean: 0.4867  Cosine: 0.2765
#   ZnO vs HgXe  Euclidean: 2.0511  Cosine: 0.8482

# Magpie (handcrafted elemental properties)
zno = CompositionalEmbedding("ZnO", "magpie_sc")
cdo = CompositionalEmbedding("CdO", "magpie_sc")
hgxe = CompositionalEmbedding("HgXe", "magpie_sc")

print("\nMagpie")
print(f"  ZnO vs CdO   Euclidean: {zno.distance(cdo, 'euclidean'):.4f}  Cosine: {zno.distance(cdo, 'cosine_distance'):.4f}")
print(f"  ZnO vs HgXe  Euclidean: {zno.distance(hgxe, 'euclidean'):.4f}  Cosine: {zno.distance(hgxe, 'cosine_distance'):.4f}")
# Magpie
#   ZnO vs CdO   Euclidean: 0.6382  Cosine: 0.0107
#   ZnO vs HgXe  Euclidean: 4.7569  Cosine: 0.7492
