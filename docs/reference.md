# Elemental Embeddings

The data contained in this folder is a collection of various elemental representation/embedding schemes

## Magpie
The following paper describes the details of the Materials Agnostic Platform for Informatics and Exploration (Magpie) framework:
[A general-purpose machine learning framework for predicting properties of inorganic materials](https://www.nature.com/articles/npjcompumats201628)

The source code for Magpie can be found
[here](https://bitbucket.org/wolverton/magpie/src/master/)

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

* `magpie_sc` is scaled version of the magpie embeddings

## mat2vec

The following paper describes the implementation of mat2vec:
[Unsupervised word embeddings capture latent knowledge from materials science literature](https://www.nature.com/articles/s41586-019-1335-8)

## MatScholar

The following paper describes the natural language processing implementation of Materials Scholar (matscholar):
[Named Entity Recognition and Normalization Applied to Large-Scale Information Extraction from the Materials Science Literature](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00470)

## MEGnet
The following paper describes the details of the construction of the MatErials Graph Network (MEGNet):
[Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://doi.org/10.1021/acs.chemmater.9b01294)

## Modified Pettifor scale
The following paper describes the details of the modified Pettifor chemical scale:
[The optimal one dimensional periodic table: a modified Pettifor chemical scale from data mining](https://iopscience.iop.org/article/10.1088/1367-2630/18/9/093011/meta)

## Oliynkyk
The following paper describes the details:
[High-Throughput Machine-Learning-Driven Synthesis of Full-Heusler Compounds](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.6b02724)

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

* `oliynyk_sc` is scaled version of the oliynyk embeddings

## Random

This is a set of 200-dimensional vectors in which the components are randomly generated

The 118 200-dimensional vectors in `random_200_new` was generated using the following code:

```python
import numpy as np

mu , sigma = 0 , 0.1 # mean and standard deviation s = np.random.normal(mu, sigma, 1000)
s = np.random.default_rng(seed=42).normal(mu, sigma, (118,200))
```
## SkipAtom

The following paper describes the details:
[Distributed representations of atoms and materials for machine learning](https://www.nature.com/articles/s41524-022-00729-3)
