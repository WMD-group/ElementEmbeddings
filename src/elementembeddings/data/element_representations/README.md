# Elemental Embeddings

The data contained in this folder is a collection of various elemental representation/embedding schemes. We provide the literature source for these representations as well as the data source for which the files were obtained. A majority of these representations have been obtained from the following repositories:

* [lrcfmd/ElMD](https://github.com/lrcfmd/ElMD/tree/master)
* [Kaaiian/CBFV](https://github.com/Kaaiian/CBFV/tree/master)

## Linear representations

For the linear/scalar representations, the `Embedding` class will load these representations as one-hot vectors where the vector components are ordered following the scale (i.e. the `atomic` representation is ordered by atomic numbers).

### Modified Pettifor scale

The following paper describes the details of the modified Pettifor chemical scale:
[The optimal one-dimensional periodic table: a modified Pettifor chemical scale from data mining](https://iopscience.iop.org/article/10.1088/1367-2630/18/9/093011/meta)

[Data source](https://github.com/lrcfmd/ElMD/blob/master/ElMD/el_lookup/mod_petti.json)

### Atomic numbers

We included `atomic` as a linear representation to generate one-hot vectors corresponding to the atomic numbers

## Vector representations

The following representations are all vector representations (some are local, some are distributed) and the `Embedding` class will load these representations as they are.

### Magpie

The following paper describes the details of the Materials Agnostic Platform for Informatics and Exploration (Magpie) framework:
[A general-purpose machine learning framework for predicting properties of inorganic materials](https://www.nature.com/articles/npjcompumats201628)

The source code for Magpie can be found
[here](https://bitbucket.org/wolverton/magpie/src/master/)

[Data source](https://github.com/Kaaiian/CBFV/blob/master/cbfv/element_properties/magpie.csv)

The 22 dimensional embedding vector includes the following elemental properties:

<details>
    <summary>Click to see the 22 properties</summary>

* Number;
* Mendeleev number;
* Atomic weight;
* Melting temperature;
* Group number;
* Period;
* Covalent Radius; 
* Electronegativity;
* no. of s, p, d, f  valence electrons (4 features);
* no. of valence electrons;
* no. of unfilled: s, p, d, f orbitals (4 features),
* no. of unfilled orbtials
* GSvolume_pa (DFT volume per atom of T=0K ground state from the OQMD)
* GSbandgap(DFT bandgap energy of T=0K ground state from the OQMD)
* GSmagmom (DFT magnetic moment of T=0K ground state from the OQMD)
* Space Group Number
</details>

* `magpie_sc` is a scaled version of the magpie embeddings. [Data source](https://github.com/lrcfmd/ElMD/blob/master/ElMD/el_lookup/magpie_sc.json)

### mat2vec

The following paper describes the implementation of mat2vec:
[Unsupervised word embeddings capture latent knowledge from materials science literature](https://www.nature.com/articles/s41586-019-1335-8)

[Data source](https://github.com/Kaaiian/CBFV/blob/master/cbfv/element_properties/mat2vec.csv)

### MatScholar

The following paper describes the natural language processing implementation of Materials Scholar (matscholar):
[Named Entity Recognition and Normalization Applied to Large-Scale Information Extraction from the Materials Science Literature](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00470)

[Data source](https://github.com/lrcfmd/ElMD/blob/master/ElMD/el_lookup/matscholar.json)

### MEGnet

The following paper describes the details of the construction of the MatErials Graph Network (MEGNet):
[Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://doi.org/10.1021/acs.chemmater.9b01294)

[Data source](https://github.com/lrcfmd/ElMD/blob/master/ElMD/el_lookup/megnet16.json)

### Oliynyk

The following paper describes the details:
[High-Throughput Machine-Learning-Driven Synthesis of Full-Heusler Compounds](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.6b02724)

[Data source](https://github.com/Kaaiian/CBFV/blob/master/cbfv/element_properties/oliynyk.csv)

We have preprocessed the original `oliynyk.csv` to account for the missing values. The preprocessed file is called `oliynyk_preprocessed.csv` which is now the default file which is called by `load_data`.

The 44 features of the embedding vector are formed of the following properties:
<details>
    <summary> Click to see the 44 features!</summary>

* Number
* Atomic_Weight
* Period
* Group
* Families
* Metal
* Nonmetal
* Metalliod
* Mendeleev_Number
* l_quantum_number
* Atomic_Radius
* Miracle_Radius_[pm]
* Covalent_Radius
* Zunger_radii_sum
* Ionic_radius
* crystal_radius
* Pauling_Electronegativity
* MB_electonegativity
* Gordy_electonegativity
* Mulliken_EN
* Allred-Rockow_electronegativity
* Metallic_valence
* Number_of_valence_electrons
* Gilmor_number_of_valence_electron
* valence_s
* valence_p
* valence_d
* valence_f
* Number_of_unfilled_s_valence_electrons
* Number_of_unfilled_p_valence_electrons
* Number_of_unfilled_d_valence_electrons
* Number_of_unfilled_f_valence_electrons
* Outer_shell_electrons
* 1st_ionization_potential_(kJ/mol)
* Polarizability(A^3)
* Melting_point_(K)
* Boiling_Point_(K)
* Density_(g/mL)
* Specific_heat_(J/g_K)_
* Heat_of_fusion_(kJ/mol)_
* Heat_of_vaporization_(kJ/mol)_
* Thermal_conductivity_(W/(m_K))_
* Heat_atomization(kJ/mol)
* Cohesive_energy
</details>

* `oliynyk_sc` is a scaled version of the oliynyk embeddings: [Data source](https://github.com/lrcfmd/ElMD/blob/master/ElMD/el_lookup/oliynyk_sc.json)

### Random

This is a set of 200-dimensional vectors in which the components are randomly generated

The 118 200-dimensional vectors in `random_200_new` were generated using the following code:

```python
import numpy as np

mu , sigma = 0 , 1 # mean and standard deviation s = np.random.normal(mu, sigma, 1000)
s = np.random.default_rng(seed=42).normal(mu, sigma, (118,200))
```

### SkipAtom

The following paper describes the details:
[Distributed representations of atoms and materials for machine learning](https://www.nature.com/articles/s41524-022-00729-3)

[Data source](https://github.com/lantunes/skipatom/blob/main/data/skipatom_20201009_induced.csv)

### CrystaLLM

The following paper describes the details behind the crystal structure generation model which uses large language modelling: [Crystal Structure Generation with Autoregressive Large Language Modeling](https://arxiv.org/abs/2307.04340)

### XenonPy

The XenonPy embedding uses the 58 features which are commonly used in publications that use the [XenonPy package](https://github.com/yoshida-lab/XenonPy).
See the following publications:
* [Representation of materials by kernel mean embedding](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.134107) 
* [Crystal structure prediction with machine learning-based element substitution](https://www.sciencedirect.com/science/article/pii/S0927025622002555)
