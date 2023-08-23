"""Setup script for AtomicEmbeddings."""
import os

from setuptools import find_namespace_packages, setup

module_dir = os.path.dirname(os.path.abspath(__file__))

VERSION = "0.4"
DESCRIPTION = "Element Embeddings"
with open(os.path.join(module_dir, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()


# Setting up
setup(
    name="ElementEmbeddings",
    version=VERSION,
    author="Anthony O. Onwuli",
    author_email="anthony.onwuli16@imperial.ac.uk",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "elementembeddings.data.element_representations": ["*.json", "*.csv"],
        "elementembeddings.data.element_data": ["*.json", "*.txt"],
    },
    test_suite="elementembeddings.tests.test",
    install_requires=[
        "numpy==1.23.3",
        "scipy==1.10.1",
        "pymatgen>2022.9.21",
        "seaborn==0.12.2",
        "matplotlib==3.7.1",
        "scikit-learn==1.3.0",
        "umap-learn==0.5.3",
        "adjustText==0.8",
    ],
    extras_require={
        "dev": [
            "pre-commit==3.3.3",
            "black==23.3.0",
            "isort==5.12.0",
            "pytest==7.2.1",
            "pytest-subtests==0.10.0",
            "nbqa==1.5.3",
            "pyupgrade==3.3.1",
            "flake8==6.0.0",
            "autopep8==2.0.2",
            "pytest-cov==4.1.0",
        ],
        "docs": [
            "mkdocs==1.4.3",
            "mkdocs-material==9.1.17",
            "mkdocstrings ==0.21.2",
            "mkdocstrings-python == 1.2.1",
            "mike ==1.1.2",
        ],
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
    ],
)
