"""Setup script for AtomicEmbeddings."""
import os

from setuptools import find_namespace_packages, setup

module_dir = os.path.dirname(os.path.abspath(__file__))

VERSION = "0.1"
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
        "elementembeddings.data": ["*.json", "*.csv"],
        "elementembeddings.data.element_data": ["*.json", "*.txt"],
    },
    test_suite="elementembeddings.tests.test",
    install_requires=[
        "numpy",
        "scipy",
        "pymatgen",
        "seaborn",
        "matplotlib",
        "scikit-learn",
        "umap-learn",
        "adjustText",
    ],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
