Here we will demonstrate how to use some of `AtomicEmbeddings`'s features. For full worked examples of using the package, please refer to the Jupyter notebooks in the [examples section of the Github repo](https://github.com/WMD-group/Atomic_Embeddings/tree/main/examples).

## Atomic_Embeddings

The `Embedding` class lies at the heart of the package. It handles elemental representation data and enables analysis and visualisation.

```py
from AtomicEmbeddings.core import Embedding 

# Load the magpie data
magpie = Embedding.load_data('magpie')

# Print out some of the properties of the Atomic_Embeddings class

# Print the dimensions of the embedding
print(f'The magpie representation has embeddings of dimension {magpie.dim} \n') 

print(magpie.element_list) # prints out all the elements considered for this representation

The magpie representation has embeddings of dimension 21
['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk']

```