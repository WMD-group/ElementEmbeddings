from setuptools import setup, find_packages

VERSION ='0.0.2'
DESCRIPTION = 'Atomic Embeddings'
LONG_DESCRIPTION = "A package for visualising and analysing atomic embedding vectors"

# Setting up
setup(
    name="AtomicEmbeddings",
    version=VERSION,
    author="Anthony O. Onwuli",
    author_email="anthony.onwuli16@imperial.ac.uk",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pymatgen",
        "seaborn",
        "matplotlib",
        "scikit-learn"
    ],
    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"

    ]
)
