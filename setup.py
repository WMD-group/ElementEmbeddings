"""Setup script for AtomicEmbeddings."""
import os

from setuptools import find_namespace_packages, setup

module_dir = os.path.dirname(os.path.abspath(__file__))

VERSION = "0.5"
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
        "numpy>=1.23.3,<2",
        "scipy>=1.10.1",
        "pymatgen>2022.9.21",
        "seaborn>=0.13.0",
        "matplotlib>=3.7.1",
        "scikit-learn>=1.3.0",
        "umap-learn>=0.5.3",
        "adjustText>=0.8",
        "openTSNE>=1.0.0",
        "typing-extensions",
    ],
    extras_require={
        "dev": [
            "pre-commit==3.7.1",
            "black==24.3.0",
            "isort==5.13.2",
            "pytest==8.2.2",
            "pytest-subtests==0.10.0",
            "nbqa==1.7.1",
            "flake8==7.1.0",
            "pyupgrade==3.13.0",
            "autopep8==2.0.2",
            "pytest-cov==5.0.0",
            "pytest-mpl==0.17.0",
        ],
        "docs": [
            "mkdocs==1.6.0",
            "mkdocs-material==9.5.16",
            "mkdocstrings ==0.25.1",
            "mkdocstrings-python == 1.9.0",
            "mike ==2.1.2",
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
