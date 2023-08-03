# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

<small>[Compare with latest](https://github.com/WMD-group/ElementEmbeddings/compare/v0.2.0...HEAD)</small>

### Added

- Add images to resources folder ([99cd979](https://github.com/WMD-group/ElementEmbeddings/commit/99cd979a51a40618e86ed4dc125b3e2c93e992b3) by Anthony Onwuli).
- Add oliynyk csv load_data method ([7fa064a](https://github.com/WMD-group/ElementEmbeddings/commit/7fa064ab6b8deb5eb516312b01a8160c7c78b565) by Anthony Onwuli).
- Add data sources to representation README ([0a27e35](https://github.com/WMD-group/ElementEmbeddings/commit/0a27e358ee62df0ec1e96dd8b869935beed60ae2) by Anthony Onwuli).
- Add csv file for oliynyk representation ([9f4de0d](https://github.com/WMD-group/ElementEmbeddings/commit/9f4de0dc407c22a6291e594dc8270cd5f27c5852) by Anthony Onwuli).
- Add method for standardising embeddings ([c3b3b65](https://github.com/WMD-group/ElementEmbeddings/commit/c3b3b6565527d4b9ec86f530626c42d3adf2f0a7) by Anthony Onwuli).
- Add class attribute for checking standardisation ([c3647e6](https://github.com/WMD-group/ElementEmbeddings/commit/c3647e682c67a1d4ca8a326d919cbd4bc487560e) by Anthony Onwuli).
- Added example of featurising a formula dataframe ([96dd41a](https://github.com/WMD-group/ElementEmbeddings/commit/96dd41ac66dbcc88ce164c0dd32d79bd6d683bb6) by Anthony Onwuli).
- Add feature labels to composition featuriser ([d8e2b64](https://github.com/WMD-group/ElementEmbeddings/commit/d8e2b64790a756896bde2803684c5c059976d3e4) by Anthony Onwuli).
- Added feature labels to Embedding class ([5e58483](https://github.com/WMD-group/ElementEmbeddings/commit/5e58483da14050fef618ac3220d3c7f62f284dc9) by Anthony Onwuli).
- Added tests for kwargs in dimension plotter ([3eba22b](https://github.com/WMD-group/ElementEmbeddings/commit/3eba22b291de218a8dda6aadd69f2a42d68034cd) by Anthony Onwuli).
- Add params argument to dimension plotter ([a041175](https://github.com/WMD-group/ElementEmbeddings/commit/a04117578d97ef3f837a87ba2ae229774048be21) by Anthony Onwuli).
- Add test for one-hot atomic number Embedding ([27da6c3](https://github.com/WMD-group/ElementEmbeddings/commit/27da6c37868be4b7b35cc5e0f0db0a7502290a19) by Anthony Onwuli).
- Add test for modified pettifor scale Embedding ([4870b25](https://github.com/WMD-group/ElementEmbeddings/commit/4870b2546ae95f6265bbc48dee60178e118064dd) by Anthony Onwuli).
- Added linear atomic number representation ([4f26c51](https://github.com/WMD-group/ElementEmbeddings/commit/4f26c51dc9fba834dece653e6d762f8e13c90bd2) by Anthony Onwuli).
- Add one-hot rep for modified petti scale ([68154da](https://github.com/WMD-group/ElementEmbeddings/commit/68154da7dc341db670e22348ec4f038b9b791ed5) by Anthony Onwuli).
- Added a link in the README.md ([bea710d](https://github.com/WMD-group/ElementEmbeddings/commit/bea710d80fd8394e949e03e60bfea8f081f79a27) by Anthony Onwuli).
- Added examples folder link to README ([2ccbc64](https://github.com/WMD-group/ElementEmbeddings/commit/2ccbc6457d303b54289fe87268bdcf6a1ad2c4ec) by Anthony Onwuli).
- Added distance example ([1d5b489](https://github.com/WMD-group/ElementEmbeddings/commit/1d5b489217256d3a54f90f61b03f889f063c98f5) by Anthony Onwuli).
- Added test for composition distance function ([116d587](https://github.com/WMD-group/ElementEmbeddings/commit/116d587d074ceddcc0cd850588c43af74d6dbceb) by Anthony Onwuli).
- Added distance function to composition class ([b827bc0](https://github.com/WMD-group/ElementEmbeddings/commit/b827bc0361f4e3e495679ccfe26eb94b9a8426df) by Anthony Onwuli).

### Fixed

- Fix package name in contributing.md ([075e657](https://github.com/WMD-group/ElementEmbeddings/commit/075e657d260aa72fca91c7f8cc57466235788426) by Anthony Onwuli).
- Fixed bug with is_standarised attribute ([ebaf773](https://github.com/WMD-group/ElementEmbeddings/commit/ebaf773134c8ab4988f071da9eca2193e4340f60) by Anthony Onwuli).
- Fixed composistion featuriser test ([3b729c9](https://github.com/WMD-group/ElementEmbeddings/commit/3b729c9584bf7cc4de2fd5c70baff41c8753800b) by Anthony Onwuli).
- Fixed dim for linear/scalar representations ([f6cc681](https://github.com/WMD-group/ElementEmbeddings/commit/f6cc681ee4f343240f5bb4d49cf9dfc5fa89f455) by Anthony Onwuli).

### Changed

- Changed load_data method ([c558480](https://github.com/WMD-group/ElementEmbeddings/commit/c558480e230d3502edd4ad9aa648a710b6782f1e) by Anthony Onwuli).
- changed default arg to a str from a list ([b5f7cd7](https://github.com/WMD-group/ElementEmbeddings/commit/b5f7cd79d9c50b0652cc02d22dedbc683f680fc2) by Anthony Onwuli).

<!-- insertion marker -->
## [v0.2.0](https://github.com/WMD-group/ElementEmbeddings/releases/tag/v0.2.0) - 2023-07-07

<small>[Compare with v0.1.1](https://github.com/WMD-group/ElementEmbeddings/compare/v0.1.1...v0.2.0)</small>

### Added

- Added dev & doc reqs and updated version no. ([e9278c5](https://github.com/WMD-group/ElementEmbeddings/commit/e9278c579a031643576f137196aad34e0f5ea98f) by Anthony Onwuli).
- Added test for dimension plotter in 3D ([ef37daf](https://github.com/WMD-group/ElementEmbeddings/commit/ef37daff6aa824c3d9917fa1ba26fc37b95a9951) by Anthony Onwuli).
- Added test for plotting euclidean heatmap ([e9cc5ed](https://github.com/WMD-group/ElementEmbeddings/commit/e9cc5ed5508e624420b1330973425572ff5b1628) by Anthony Onwuli).
- Added deprecation warning to old plotting function ([542e2f2](https://github.com/WMD-group/ElementEmbeddings/commit/542e2f2e6bd96b0f0e1624192cb9a9a98fb3dfcc) by Anthony Onwuli).
- Added functions to test PCA and UMAP for Embedding ([b7ccc8f](https://github.com/WMD-group/ElementEmbeddings/commit/b7ccc8f41384e5e6095090aa016088279b5a0439) by Anthony Onwuli).
- Added test function for tSNE for Embedding class ([aaa1472](https://github.com/WMD-group/ElementEmbeddings/commit/aaa147279ba609984482813df2ce9530420da2be) by Anthony Onwuli).
- Added more tests for compute metrics function ([c4055bc](https://github.com/WMD-group/ElementEmbeddings/commit/c4055bcdad6e5bd7832a8568767ced72cd9cdfd9) by Anthony Onwuli).
- Added test for computing spearman's rank ([5f715aa](https://github.com/WMD-group/ElementEmbeddings/commit/5f715aaa3ba339b5e01012cb0a40c44652481b55) by Anthony Onwuli).
- Added more dataframe test functions ([edeeb87](https://github.com/WMD-group/ElementEmbeddings/commit/edeeb8714ae80b194159738b562606819ffc3ccb) by Anthony Onwuli).
- Added test for removing multiple elements ([b69d806](https://github.com/WMD-group/ElementEmbeddings/commit/b69d80699cad211166ff1b112886d19d387890b5) by Anthony Onwuli).
- Added test function for removing elements ([664cbbf](https://github.com/WMD-group/ElementEmbeddings/commit/664cbbf1846757b7d018c199745b6227465c0268) by Anthony Onwuli).
- Added setUpClass to test_core.py ([7cb8ab6](https://github.com/WMD-group/ElementEmbeddings/commit/7cb8ab6d3b731d04831cdfe83a90b926ab1e2a1b) by Anthony Onwuli).
- Added tests for `as_dataframe` method ([7256c23](https://github.com/WMD-group/ElementEmbeddings/commit/7256c23d8d2840b77983424ee9247a90f1caaded) by Anthony Onwuli).
- Added more tests to the Embedding loading function ([7f9f87b](https://github.com/WMD-group/ElementEmbeddings/commit/7f9f87b987a77f1d4b73cf9fed289a5d9a028417) by Anthony Onwuli).
- Added test functions for loading csv and json ([87ba958](https://github.com/WMD-group/ElementEmbeddings/commit/87ba9581c506bc16aa377961da28c0cbe60e80de) by Anthony Onwuli).
- Added test embedding json file ([188aea4](https://github.com/WMD-group/ElementEmbeddings/commit/188aea48e21b3c3d5a1b9624a9885b94f14b2fcc) by Anthony Onwuli).
- Added test embedding csv file ([173bcee](https://github.com/WMD-group/ElementEmbeddings/commit/173bcee057173ec1a48cdc7bb3141406236119ce) by Anthony Onwuli).

### Fixed

- Fixed spelling error for extras_require setup.py ([8e28e9a](https://github.com/WMD-group/ElementEmbeddings/commit/8e28e9a09550bfcaf21ec4d95989cd031d717596) by Anthony Onwuli).

### Removed

- Removed outdated installation instructions ([c69817f](https://github.com/WMD-group/ElementEmbeddings/commit/c69817fef331e203fb3861e603c7c0176097e51f) by Anthony Onwuli).
- Removed an else block from `load_data` ([5532de6](https://github.com/WMD-group/ElementEmbeddings/commit/5532de6d050580382f0fa9688be96f0e9cd231ec) by Anthony Onwuli).

## [v0.1.1](https://github.com/WMD-group/ElementEmbeddings/releases/tag/v0.1.1) - 2023-07-05

<small>[Compare with v0.1](https://github.com/WMD-group/ElementEmbeddings/compare/v0.1...v0.1.1)</small>

### Added

- added pytest-subtests to dev reqs ([03f3108](https://github.com/WMD-group/ElementEmbeddings/commit/03f31088d5be656f9fe67d88bd850a1817bd862d) by Anthony Onwuli).
- added doc and pypi badges to README ([8ea476c](https://github.com/WMD-group/ElementEmbeddings/commit/8ea476cf1422ca0d94d795fcd7b58ebd0ea858fe) by Anthony Onwuli).
- Added citation file ([8d0baa1](https://github.com/WMD-group/ElementEmbeddings/commit/8d0baa1cd17b787e465300452d3c2d16a56c009b) by Anthony Onwuli).

### Removed

- removed pandas, pytest and pytest-subtests in reqs ([cd1bf77](https://github.com/WMD-group/ElementEmbeddings/commit/cd1bf776220250377bb7cd48cca6b08e9a968f1d) by Anthony Onwuli).

## [v0.1](https://github.com/WMD-group/ElementEmbeddings/releases/tag/v0.1) - 2023-06-30

<small>[Compare with first commit](https://github.com/WMD-group/ElementEmbeddings/compare/262c7e99a438a3527fb73866093ae8cb1ee85ee6...v0.1)</small>

### Added

- added documentation ([a9264f4](https://github.com/WMD-group/ElementEmbeddings/commit/a9264f41035e8b6bdeeb2255ef0f9743a7d1be19) by Anthony Onwuli).
- Added scaled magpie embedding ([eb33ab4](https://github.com/WMD-group/ElementEmbeddings/commit/eb33ab4921343889f7583abe11eccdc2f34d8ffd) by Anthony Onwuli).
- Added tests for the plotting module ([6e1098d](https://github.com/WMD-group/ElementEmbeddings/commit/6e1098db830e8168ab3f65a5e4b50ed0bf8221b2) by Anthony Onwuli).
- added pytest-cov to dev reqs ([6db996f](https://github.com/WMD-group/ElementEmbeddings/commit/6db996f9981fa991ffb1435dec4d0a6ef3ec6544) by Anthony Onwuli).
- added adjustText to requirements ([6ebd9ad](https://github.com/WMD-group/ElementEmbeddings/commit/6ebd9ad45b6e178904ae8c3e9fa4e260fb0012f1) by Anthony Onwuli).
- added scaling to dimension reduction calcs ([3f177d8](https://github.com/WMD-group/ElementEmbeddings/commit/3f177d89f1aca2406372ff62f750d13ac3ec1c26) by Anthony Onwuli).
- added function for plotting reduced Dims ([4085b94](https://github.com/WMD-group/ElementEmbeddings/commit/4085b948f46e2cd490d7dec9d309513f2ad7c69e) by Anthony).
- added DR functions ([f755dc8](https://github.com/WMD-group/ElementEmbeddings/commit/f755dc870d9e494fd2dcac9d748d7a46e8844db0) by Anthony).
- Added similarity plots in notebook ([0118932](https://github.com/WMD-group/ElementEmbeddings/commit/011893228b6aba848df060796a30696de7db4a7a) by Anthony Onwuli).
- Added new random embedding for all 118 elements ([c6e9d8b](https://github.com/WMD-group/ElementEmbeddings/commit/c6e9d8bfecfa68fad37f47146fbeb43e06489c1c) by Anthony).
- added axis sorting argument to plotter function ([d1a11b0](https://github.com/WMD-group/ElementEmbeddings/commit/d1a11b04cf8a32cc8164b727b71aaac27af8883a) by Anthony).
- added sorting options to distance methods ([1947bb5](https://github.com/WMD-group/ElementEmbeddings/commit/1947bb58c00ba78842bfe237a1bff822476cb7e0) by Anthony).
- added deprecation warnings ([fcc53a2](https://github.com/WMD-group/ElementEmbeddings/commit/fcc53a23772ce1e2bfd9d450e5530e2e3030f123) by Anthony).
- added files for periodic table information ([ea2e40a](https://github.com/WMD-group/ElementEmbeddings/commit/ea2e40ac045ce620d025de2fc756bab6a22fcb19) by Anthony).
- Added composition featuriser to the example ([a16b780](https://github.com/WMD-group/ElementEmbeddings/commit/a16b7802a9ab243b3766de54e2bfafd6f1ee0aca) by Anthony).
- Added a multi plotter function ([e26f31c](https://github.com/WMD-group/ElementEmbeddings/commit/e26f31cac0366f259e92b0c4abbe49eeddcc2524) by Anthony).
- added tests for utils module ([eefa232](https://github.com/WMD-group/ElementEmbeddings/commit/eefa232b8bad56e2f662f7898732de94fbcffa06) by Anthony).
- added cosine measures to core.py ([e75af55](https://github.com/WMD-group/ElementEmbeddings/commit/e75af558b930c66204b05882e333a81fc34e8bbd) by Anthony).
- added hook to remove nb outputs in pre-commit ([6d9c32d](https://github.com/WMD-group/ElementEmbeddings/commit/6d9c32dc6d5c19e4e0259f1c2e978573369e2485) by Anthony).
- Added docstrings to utils/io.py ([962f79a](https://github.com/WMD-group/ElementEmbeddings/commit/962f79a4c4936f322a3fe5fed23a4789f261283d) by Anthony).
- Added docstrings to test_core.py ([eddc26e](https://github.com/WMD-group/ElementEmbeddings/commit/eddc26e31468e89c324873352736c1e7f8f61eb1) by Anthony).
- Added flake8 ([eebbc8a](https://github.com/WMD-group/ElementEmbeddings/commit/eebbc8a5450f6239df76a01b364fb0583ace5e60) by Anthony).
- added progress bars to composition_featuriser ([bf9c608](https://github.com/WMD-group/ElementEmbeddings/commit/bf9c60894ca57f8552940a7c17cce3bc6c63ae19) by Anthony).
- added composition featurizer ([e116efa](https://github.com/WMD-group/ElementEmbeddings/commit/e116efa3ba4b09087a70cebebee377f66fec4f2a) by Anthony).
- added codecov badge to the main README ([2eb93a9](https://github.com/WMD-group/ElementEmbeddings/commit/2eb93a9af7c58688e49a98b121562f1ef3732e6f) by Anthony).
- added stats functions to composition.py ([e11bec3](https://github.com/WMD-group/ElementEmbeddings/commit/e11bec301a0a504b693be1d5c69b88e6ec0a1b16) by Anthony).
- add dependabot bot to repo ([3139fba](https://github.com/WMD-group/ElementEmbeddings/commit/3139fba592bc7acc810714a5bf082265b3e46244) by Anthony).
- added .svg files to gitignore ([1129e71](https://github.com/WMD-group/ElementEmbeddings/commit/1129e71296c20d227f739b82d7b5383fdaa34c75) by Anthony).
- added jupyter qa to pre-commit ([17796f5](https://github.com/WMD-group/ElementEmbeddings/commit/17796f5d34ed3fc6f5dd439982ad745b008934f3) by Anthony).
- added new IO options to Embedding class ([af219f0](https://github.com/WMD-group/ElementEmbeddings/commit/af219f0252caaf13ac1fad2e1e76091114c19e9c) by Anthony).
- added a Refactoring document ([e885382](https://github.com/WMD-group/ElementEmbeddings/commit/e885382e70679862f7bbaba9ace1855c34f7dcb6) by Anthony).
- Added qa to Github actions ([42f478c](https://github.com/WMD-group/ElementEmbeddings/commit/42f478cc5ef5b8c5e497366de37e99554ea25674) by Anthony).
- added test modules ([1864e1d](https://github.com/WMD-group/ElementEmbeddings/commit/1864e1d55ad56d9aaed9fa176c7904124779ef72) by Anthony).
- added scaled versions of magpie and oliynyk ([9418c7c](https://github.com/WMD-group/ElementEmbeddings/commit/9418c7c0afb644e4f7e3424f4bb06bd8ca605517) by Anthony).
- added a README in the data folder ([5568057](https://github.com/WMD-group/ElementEmbeddings/commit/55680575e46bc4c361fcf7714c8f375e22d5c084) by Anthony).
- added new element representations ([bb7c19e](https://github.com/WMD-group/ElementEmbeddings/commit/bb7c19ef6cb5218a84414981b92bfee309b4a880) by Anthony).
- added matscholar embeddings in data dir ([e49c8a4](https://github.com/WMD-group/ElementEmbeddings/commit/e49c8a4b2bcbc084d3753e734b416dfc12beceac) by Anthony).
- added README.md to the repo ([262c7e9](https://github.com/WMD-group/ElementEmbeddings/commit/262c7e99a438a3527fb73866093ae8cb1ee85ee6) by Anthony).

### Fixed

- Fix error with using spearman ([7afb186](https://github.com/WMD-group/ElementEmbeddings/commit/7afb1865069648e8554e0ebd385696358c694833) by Anthony Onwuli).
- fixed dimension plotter and added adjusttext ([ccca970](https://github.com/WMD-group/ElementEmbeddings/commit/ccca970d435d96499f62f69d74560cb523a39033) by Anthony Onwuli).
- Fixed code block in file ([ef524e7](https://github.com/WMD-group/ElementEmbeddings/commit/ef524e7b91f90ca9db15313fc41e2c6d9b2a781b) by Anthony).
- Fixed formatting on setup.py ([8be3961](https://github.com/WMD-group/ElementEmbeddings/commit/8be396121e47e4bd8bca605aff98eaf095bd4608) by Anthony).
- fixed typing errors ([5b005f6](https://github.com/WMD-group/ElementEmbeddings/commit/5b005f64d8513e941122b8b594739e1d84b270bb) by Anthony).
- fixed indentation error in Atomic_Embeddings class ([b7e39ab](https://github.com/WMD-group/ElementEmbeddings/commit/b7e39ab88ebc4807ef276c50be36e7c27ac47d5f) by Anthony).

### Removed

- remove unused method ([46d20af](https://github.com/WMD-group/ElementEmbeddings/commit/46d20af77797e452152a2af139e72b4d9f6145c4) by Anthony Onwuli).
- removed heatmap image ([868fe92](https://github.com/WMD-group/ElementEmbeddings/commit/868fe92c3d41b60fc6b988273e2a1ac1e5065f13) by Anthony).
- removed backup pyproject and setup files ([33f2cce](https://github.com/WMD-group/ElementEmbeddings/commit/33f2cce0e341f6837641c98cba29c22de2942252) by Anthony).
- removed numba from requirements.txt ([1eb833e](https://github.com/WMD-group/ElementEmbeddings/commit/1eb833ec1769ead345de23e2c09c92cee72cc545) by Anthony).
- removed the old AtomicEmbeddings.py file ([9b1b2e8](https://github.com/WMD-group/ElementEmbeddings/commit/9b1b2e82e0426136020f5112d12c18051d9b7c30) by Anthony).
- removed itertools from reqssetup.py ([c9c21ca](https://github.com/WMD-group/ElementEmbeddings/commit/c9c21ca8a8e242fc99f19d1fee204da7a1bbdc62) by Anthony).

