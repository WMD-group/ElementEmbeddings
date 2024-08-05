"""Configuration variables for ElementEmbeddings."""
DEFAULT_ELEMENT_EMBEDDINGS = {
    "magpie": "magpie.csv",
    "magpie_sc": "magpie_sc.json",
    "mat2vec": "mat2vec.csv",
    "matscholar": "matscholar-embedding.json",
    "megnet16": "megnet16.json",
    "mod_petti": "mod_petti.json",
    "oliynyk": "oliynyk_preprocessed.csv",
    "oliynyk_sc": "oliynyk_sc.json",
    "random_200": "random_200_new.csv",
    "skipatom": "skipatom_20201009_induced.csv",
    "atomic": "atomic.json",
    "crystallm": "crystallm_v24c.dim512_atom_vectors.csv",
    "xenonpy": "xenonpy_element_features.csv",
    "cgnf": "cgnf.json",
}

DEFAULT_SPECIES_EMBEDDINGS = {
    "skipspecies": "skipspecies_2022_10_28_dim200.csv",
    "skipspecies_induced": "skipspecies_2022_10_28_induced_dim200.csv",
}
CITATIONS = {
    "magpie": [
        "@article{ward2016general,"
        "title={A general-purpose machine learning framework for "
        "predicting properties of inorganic materials},"
        "author={Ward, Logan and Agrawal, Ankit and Choudhary, Alok "
        "and Wolverton, Christopher},"
        "journal={npj Computational Materials},"
        "volume={2},"
        "number={1},"
        "pages={1--7},"
        "year={2016},"
        "publisher={Nature Publishing Group}}",
    ],
    "magpie_sc": [
        "@article{ward2016general,"
        "title={A general-purpose machine learning framework for "
        "predicting properties of inorganic materials},"
        "author={Ward, Logan and Agrawal, Ankit and Choudhary, Alok "
        "and Wolverton, Christopher},"
        "journal={npj Computational Materials},"
        "volume={2},"
        "number={1},"
        "pages={1--7},"
        "year={2016},"
        "publisher={Nature Publishing Group}}",
    ],
    "mat2vec": [
        "@article{tshitoyan2019unsupervised,"
        "title={Unsupervised word embeddings capture latent knowledge "
        "from materials science literature},"
        "author={Tshitoyan, Vahe and Dagdelen, John and Weston, Leigh "
        "and Dunn, Alexander and Rong, Ziqin and Kononova, Olga "
        "and Persson, Kristin A and Ceder, Gerbrand and Jain, Anubhav},"
        "journal={Nature},"
        "volume={571},"
        "number={7763},"
        "pages={95--98},"
        "year={2019},"
        "publisher={Nature Publishing Group} }",
    ],
    "matscholar": [
        "@article{weston2019named,"
        "title={Named entity recognition and normalization applied to "
        "large-scale information extraction from the materials "
        "science literature},"
        "author={Weston, Leigh and Tshitoyan, Vahe and Dagdelen, John and "
        "Kononova, Olga and Trewartha, Amalie and Persson, Kristin A and "
        "Ceder, Gerbrand and Jain, Anubhav},"
        "journal={Journal of chemical information and modeling},"
        "volume={59},"
        "number={9},"
        "pages={3692--3702},"
        "year={2019},"
        "publisher={ACS Publications} }",
    ],
    "megnet16": [
        "@article{chen2019graph,"
        "title={Graph networks as a universal machine learning framework "
        "for molecules and crystals},"
        "author={Chen, Chi and Ye, Weike and Zuo, Yunxing and "
        "Zheng, Chen and Ong, Shyue Ping},"
        "journal={Chemistry of Materials},"
        "volume={31},"
        "number={9},"
        "pages={3564--3572},"
        "year={2019},"
        "publisher={ACS Publications} }",
    ],
    "oliynyk": [
        "              @article{oliynyk2016high,"
        "title={High-throughput machine-learning-driven synthesis "
        "of full-Heusler compounds},"
        "author={Oliynyk, Anton O and Antono, Erin and Sparks, Taylor D and "
        "Ghadbeigi, Leila and Gaultois, Michael W and "
        "Meredig, Bryce and Mar, Arthur},"
        "journal={Chemistry of Materials},"
        "volume={28},"
        "number={20},"
        "pages={7324--7331},"
        "year={2016},"
        "publisher={ACS Publications} }",
    ],
    "oliynyk_sc": [
        "              @article{oliynyk2016high,"
        "title={High-throughput machine-learning-driven synthesis "
        "of full-Heusler compounds},"
        "author={Oliynyk, Anton O and Antono, Erin and Sparks, Taylor D and "
        "Ghadbeigi, Leila and Gaultois, Michael W and "
        "Meredig, Bryce and Mar, Arthur},"
        "journal={Chemistry of Materials},"
        "volume={28},"
        "number={20},"
        "pages={7324--7331},"
        "year={2016},"
        "publisher={ACS Publications} }",
    ],
    "skipatom": [
        "@article{antunes2022distributed,"
        "title={Distributed representations of atoms and materials "
        "for machine learning},"
        "author={Antunes, Luis M and Grau-Crespo, Ricardo and Butler, Keith T},"
        "journal={npj Computational Materials},"
        "volume={8},"
        "number={1},"
        "pages={1--9},"
        "year={2022},"
        "publisher={Nature Publishing Group} }",
    ],
    "mod_petti": [
        "@article{glawe2016optimal,"
        "title={The optimal one dimensional periodic table: "
        "a modified Pettifor chemical scale from data mining},"
        "author={Glawe, Henning and Sanna, Antonio and Gross, "
        "EKU and Marques, Miguel AL},"
        "journal={New Journal of Physics},"
        "volume={18},"
        "number={9},"
        "pages={093011},"
        "year={2016},"
        "publisher={IOP Publishing} }",
    ],
    "crystallm": [
        "@article{antunes2023crystal,"
        "title={Crystal structure generation "
        "with autoregressive large language modeling},"
        "author={Antunes, Luis M and Butler, Keith T and Grau-Crespo, Ricardo},"
        "journal={arXiv preprint arXiv:2307.04340},"
        "year={2023}}",
    ],
    "xenonpy": [
        "@article{liu2021machine,"
        "title={Machine learning to predict quasicrystals from chemical compositions},"
        "author={Liu, Chang and Fujita, Erina and "
        "Katsura, Yukari and Inada, Yuki and Ishikawa, Asuka and "
        "Tamura, Ryuji and Kimura, Kaoru and Yoshida, Ryo},"
        "journal={Advanced Materials},"
        "volume={33},"
        "number={36},"
        "pages={2102507},"
        "year={2021},"
        "publisher={Wiley Online Library}"
        "}",
        "@article{kusaba2022crystal,"
        "title={Crystal structure prediction with machine "
        "learning-based element substitution},"
        "author={Kusaba, Minoru and Liu, Chang and Yoshida, Ryo},"
        "journal={Computational Materials Science},"
        "volume={211},"
        "pages={111496},"
        "year={2022},"
        "publisher={Elsevier}"
        "}",
        "@article{kusaba2023representation,"
        "title={Representation of materials by kernel mean embedding},"
        "author={Kusaba, Minoru and Hayashi, Yoshihiro and "
        "Liu, Chang and Wakiuchi, Araki and Yoshida, Ryo},"
        "journal={Physical Review B},"
        "volume={108},"
        "number={13},"
        "pages={134107},"
        "year={2023},"
        "publisher={APS}"
        "}",
    ],
    "cgnf": [
        "@article{jang2024synthesizability,"
        "title={Synthesizability of materials stoichiometry "
        "using semi-supervised learning},"
        "author={Jang, Jidon and Noh, Juhwan and Zhou, Lan "
        "and Gu, Geun Ho and Gregoire, John M and Jung, Yousung},"
        "journal={Matter},"
        "volume={7},"
        "number={6},"
        "pages={2294--2312},"
        "year={2024}",
    ],
    "skipspecies": [
        "@article{Onwuli_Butler_Walsh_2024, "
        "title={Ionic species representations for materials informatics}, "
        "DOI={10.26434/chemrxiv-2024-8621l}, "
        "journal={ChemRxiv}, "
        "author={Onwuli, Anthony and Butler, Keith T. and Walsh, Aron}, year={2024}} "
        "This content is a preprint and has not been peer-reviewed.",
        "@article{antunes2022distributed,"
        "title={Distributed representations of atoms and materials "
        "for machine learning},"
        "author={Antunes, Luis M and Grau-Crespo, Ricardo and Butler, Keith T},"
        "journal={npj Computational Materials},"
        "volume={8},"
        "number={1},"
        "pages={1--9},"
        "year={2022},"
        "publisher={Nature Publishing Group} }",
    ],
    "skipspecies_induced": [
        "@article{Onwuli_Butler_Walsh_2024, "
        "title={Ionic species representations for materials informatics}, "
        "DOI={10.26434/chemrxiv-2024-8621l}, "
        "journal={ChemRxiv}, "
        "author={Onwuli, Anthony and Butler, Keith T. and Walsh, Aron}, year={2024}} "
        "This content is a preprint and has not been peer-reviewed.",
        "@article{antunes2022distributed,"
        "title={Distributed representations of atoms and materials "
        "for machine learning},"
        "author={Antunes, Luis M and Grau-Crespo, Ricardo and Butler, Keith T},"
        "journal={npj Computational Materials},"
        "volume={8},"
        "number={1},"
        "pages={1--9},"
        "year={2022},"
        "publisher={Nature Publishing Group} }",
    ],
}

ELEMENT_GROUPS_PALETTES = {
    "Alkali": "tab:blue",
    "Alkaline": "tab:cyan",
    "Lanthanoid": "tab:purple",
    "TM": "tab:orange",
    "Post-TM": "tab:green",
    "Metalloid": "tab:pink",
    "Halogen": "tab:red",
    "Noble gas": "tab:olive",
    "Chalcogen": "tab:brown",
    "Others": "tab:gray",
    "Actinoid": "thistle",
}
