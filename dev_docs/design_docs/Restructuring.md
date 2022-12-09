# Code Refactoring doc

## Motivation

`Atomic_Embeddings0.0.3` has been utilised in two 2022 MSc thesis projects and in a current unpublished work. The intentions were to provide a package which collates several elemental representations and provide high level tools for both quantitative and qualitative analysis through statistical analysis and visualisations.
To carry this package further, we need to be able develop methods which allow us to quantitatively compare the visualisations and extend the codebase to incorporate composition

## Required features

* A core module which contains the `Embedding` class
    * [**Implemented**]Has a `load_data` method to load in default embeddings distributed with the package
    * [**WIP**]Other file I/O methods to allow users to create `Embedding` instances from their own files
    * [**Implemented**] A `citations` property, so that BibTex citations are provided for the distributed representations. 
    * 

* A `composition` module
    * Required to interface with the `Embedding` class to generate composition-based feature vectors (CBFVs) based of the elemental representations
    * Should have options to define the set of operations which are used to generate the CBFVs
    * Should have methods to calculate the similarity between multiple CBFVs
    * Should have functions for parsing formula [WIP]

* Plotting utilities
    * This can either be in the form of methods to other class instances or a separate module to handle the plotting.
    * Could look to have a Class which can perform both UMAP, PCA, (t-SNE) dimensionality reductions (DRs) and another class for plotting the results. The class option would allow DRs to be done to both elemental `Embedding` objects and lists of `CompositionalEmbedding` objects.
