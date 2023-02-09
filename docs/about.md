# About the AtomicEmbeddings package
====

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub issues](https://img.shields.io/github/issues-raw/WMD-Group/Atomic_Embeddings)](https://github.com/WMD-group/Atomic_Embeddings/issues)
[![CI Status](https://github.com/WMD-group/Atomic_Embeddings/actions/workflows/ci.yml/badge.svg)](https://github.com/WMD-group/Atomic_Embeddings/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/WMD-group/Atomic_Embeddings/branch/main/graph/badge.svg?token=OCMIM5SHL0)](https://codecov.io/gh/WMD-group/Atomic_Embeddings)

The **Atomic Embeddings** package provides high-level tools for analysing elemental
embeddings data. This primarily involves visualising the correlation between
embedding schemes using different statistical measures.

Motivation
--------

Machine learning approaches for materials informatics have become increasingly
widespread. Some of these involve the use of deep learning
techniques where the representation of the elements is learned
rather than specified by the user of the model. While an important goal of
machine learning training is to minimise the chosen error function to make more
accurate predictions, it is also important for us material scientists to be able
to interpret these models. As such, we aim to evaluate and compare different atomic embedding
schemes in a consistent framework.