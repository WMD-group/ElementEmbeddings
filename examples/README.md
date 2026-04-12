# ElementEmbeddings Examples

A collection of examples using [ElementEmbeddings](https://github.com/WMD-group/ElementEmbeddings)

## Jupyter Notebooks

- **usage.ipynb**: A walkthrough of the basic features of the `Embedding` class
- **composition.ipynb**: A walkthrough of using the `ElementEmbeddings.composition` module
- **species.ipynb**: A walkthrough of using the species features present in the `ElementEmbeddings` package

## Visualisation Scripts

- **generate_cosine_heatmaps.py**: Generates cosine similarity heatmaps for all embedding schemes. Outputs static PNGs and an animated GIF to `2d_cosine/`.
- **3d_embedding_maps.py**: Generates 3D scatter plots of elements in reduced embedding space (UMAP, PCA, t-SNE) for multiple embeddings. Outputs to `3d_maps/`.
- **3d_embedding_composite.py**: Generates a composite image showing high-dimensional embedding vectors alongside a 3D reduced map. Outputs to `3d_maps/`.
