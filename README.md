# ElementEmbeddings


[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub issues](https://img.shields.io/github/issues-raw/WMD-Group/ElementEmbeddings)](https://github.com/WMD-group/ElementEmbeddings/issues)
[![CI Status](https://github.com/WMD-group/ElementEmbeddings/actions/workflows/ci.yml/badge.svg)](https://github.com/WMD-group/ElementEmbeddings/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/WMD-group/ElementEmbeddings/branch/main/graph/badge.svg?token=OCMIM5SHL0)](https://codecov.io/gh/WMD-group/ElementEmbeddings)
[![DOI](https://zenodo.org/badge/493285385.svg)](https://zenodo.org/badge/latestdoi/493285385)
[![PyPI](https://img.shields.io/pypi/v/ElementEmbeddings)](https://pypi.org/project/ElementEmbeddings/)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://wmd-group.github.io/ElementEmbeddings/)
![python version](https://img.shields.io/pypi/pyversions/elementembeddings)

The **Element Embeddings** package provides high-level tools for analysing elemental
embeddings data. This primarily involves visualising the correlation between
embedding schemes using different statistical measures.

* **Documentation:** <https://wmd-group.github.io/ElementEmbeddings/>
* **Examples:** <https://github.com/WMD-group/ElementEmbeddings/tree/main/examples>

## Motivation

Machine learning approaches for materials informatics have become increasingly
widespread. Some of these involve the use of deep learning
techniques where the representation of the elements is learned
rather than specified by the user of the model. While an important goal of
machine learning training is to minimise the chosen error function to make more
accurate predictions, it is also important for us material scientists to be able
to interpret these models. As such, we aim to evaluate and compare different atomic embedding
schemes in a consistent framework.

## Getting started

ElementEmbeddings's main feature, the Embedding class is accessible by
importing the class.

## Installation

The latest stable release can be installed via pip using:

```bash
pip install ElementEmbeddings
```

For installing the development or documentation dependencies via pip:

```bash
pip install "ElementEmbeddings[dev]"
pip install "ElementEmbeddings[docs]"
```

For development, you can clone the repository and install the package in editable mode.
To clone the repository and make a local installation, run the following commands:

```bash
git clone https://github.com/WMD-group/ElementEmbeddings.git
cd ElementEmbeddings
pip install  -e .
```

With -e pip will create links to the source folder so that changes to the code will be immediately reflected on the PATH.

## Usage

For simple usage, you can instantiate an Embedding object using one of the embeddings in the [data directory](src/elementembeddings/data/element_representations/README.md). For this example, let's use the magpie elemental representation.

```python
# Import the class
>>> from elementembeddings.core import Embedding

# Load the magpie data
>>> magpie = Embedding.load_data('magpie')
```

We can access some of the properties of the `Embedding` class. For example, we can find the dimensions of the elemental representation and the list of elements for which an embedding exists.

```python
# Print out some of the properties of the ElementEmbeddings class
>>> print(f'The magpie representation has embeddings of dimension {magpie.dim}') 
>>> print(f'The magpie representation contains these elements: \n {magpie.element_list}') # prints out all the elements considered for this representation
>>> print(f'The magpie representation contains these features: \n {magpie.feature_labels}') # Prints out the feature labels of the chosen representation

The magpie representation has embeddings of dimension 22
The magpie representation contains these elements:
['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk']
The magpie representation contains these features:
['Number', 'MendeleevNumber', 'AtomicWeight', 'MeltingT', 'Column', 'Row', 'CovalentRadius', 'Electronegativity', 'NsValence', 'NpValence', 'NdValence', 'NfValence', 'NValence', 'NsUnfilled', 'NpUnfilled', 'NdUnfilled', 'NfUnfilled', 'NUnfilled', 'GSvolume_pa', 'GSbandgap', 'GSmagmom', 'SpaceGroupNumber']
```

### Plotting

We can quickly generate heatmaps of distance/similarity measures between the element vectors using `heatmap_plotter` and plot the representations in two dimensions using the `dimension_plotter` from the plotter module. Before we do that, we will standardise the embedding using the `standardise` method available to the Embedding class

```python
from elementembeddings.plotter import heatmap_plotter, dimension_plotter
import matplotlib.pyplot as plt

magpie.standardise(inplace=True) # Standardises the representation

fig, ax = plt.subplots(1, 1, figsize=(6,6))
heatmap_params = {"vmin": -1, "vmax": 1}
heatmap_plotter(embedding=magpie, metric="cosine_similarity",show_axislabels=False,cmap="Blues_r",ax=ax, **heatmap_params)
ax.set_title("Magpie cosine similarities")
fig.tight_layout()
fig.show()

```

<img src="resources/magpie_cosine_sim_heatmap.png" alt = "Cosine similarity heatmap of the magpie representation" width="50%"/>

```python
fig, ax = plt.subplots(1, 1, figsize=(6,6))

reducer_params={"n_neighbors": 30, "random_state":42}
scatter_params = {"s":100}

dimension_plotter(embedding=magpie, reducer="umap",n_components=2,ax=ax,adjusttext=True,reducer_params=reducer_params, scatter_params=scatter_params)
ax.set_title("Magpie UMAP (n_neighbours=30)")
ax.legend().remove()
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.25, 0.5), loc="center right", ncol=1)

fig.tight_layout()
fig.show()
```

<img src="resources/magpie_umap.png" alt="Scatter plot of the Magpie representation reduced to 2 dimensions using UMAP" width="50%"/>

### Compositions

The package can also be used to featurise compositions. Your data could be a list of formula strings or a pandas dataframe of the following format:

| formula |
|---------|
| CsPbI3  |
| Fe2O3   |
| NaCl    |
| ZnS     |

The `composition_featuriser` function can be used to featurise the data. The compositions can be featurised using different representation schemes and different types of pooling through the `embedding` and `stats` arguments respectively.

```python
from elementembeddings.composition import composition_featuriser

df_featurised = composition_featuriser(df, embedding="magpie", stats="mean")

df_featurised
```

| formula | mean_Number | mean_MendeleevNumber | mean_AtomicWeight  | mean_MeltingT     | ... | mean_SpaceGroupNumber |
|---------|-------------|----------------------|--------------------|-------------------|-----|-----------------------|
| CsPbI3  | 59.2        | 74.8                 | 144.16377238       | 412.55            | ... | 129.20000000000002    |
| Fe2O3   | 15.2        | 74.19999999999999    | 31.937640000000002 | 757.2800000000001 | ... | 98.80000000000001     |
| NaCl    | 14.0        | 48.0                 | 29.221384640000004 | 271.235           | ... | 146.5                 |
| ZnS     | 23.0        | 78.5                 | 48.7225            | 540.52            | ... | 132.0                 |

(The columns of the resulting dataframe have been truncated for clarity.)

The returned dataframe contains the mean-pooled features of the magpie representation for the four formulas.

## Development notes

### Bugs, features and questions

Please use the [issue tracker](https://github.com/WMD-group/ElementEmbeddings/issues) to report bugs and any feature requests. Hopefully, most questions should be solvable through [the docs](https://wmd-group.github.io/ElementEmbeddings/). For any other queries related to the project, please contact Anthony Onwuli by [e-mail: anthony.onwuli16@imperial.ac.uk](anthony.onwuli16@imperial.ac.uk).

### Code contributions

We welcome new contributions to this project. See [the contributing guide](contributing.md) for detailed instructions on how to contribute to our project.

### Developer

* [Anthony Onwuli](https://github.com/AntObi) (Department of Materials, Imperial College London)
