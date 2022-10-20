This part of the project documentation provides the python API for the `AtomicEmbeddings` package.

## Default embeddings

The table below shows the available elemental representations within the package. These can be called using `Atomic_Embeddings.from_json(str_name)`.


| **Name**                | **str_name** |
|-------------------------|--------------|
| Magpie                  | magpie       |
| Magpie (scaled)         | magpie_sc    |
| Mat2Vec                 | mat2vec      |
| Matscholar              | matscholar   |
| Megnet (16 dimensions)  | megnet16     |
| Modified pettifor scale | mod_petti    |
| Oliynyk                 | oliynyk      |
| Oliynyk (scaled)        | oliynyk_sc   |
| Random (200 dimensions) | random_200   |
| SkipAtom                | skipatom     |

::: AtomicEmbeddings.AtomicEmbeddings
