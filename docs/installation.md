# Getting Started

The latest stable release can be installed via pip or uv:

```bash
pip install ElementEmbeddings
uv pip install ElementEmbeddings
```

## Developer's installation (optional)

For development work, `ElementEmbeddings` can be installed from a copy of the [source repository](https://github.com/WMD-group/ElementEmbeddings.git); this is preferred if using experimental code branches.

To clone the project from GitHub and create the local development environment with `uv`:

```bash
git clone https://github.com/WMD-group/ElementEmbeddings.git
cd ElementEmbeddings
uv sync --extra dev --extra docs
```

This creates `.venv` and installs the project in editable mode, so local source changes are reflected immediately.
