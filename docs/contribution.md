# Contributing

This is a quick guide on how to follow best practice and contribute smoothly to `ElementEmbeddings`.

## Code contributions

We are always looking for ways to make `ElementEmbeddings` better and a more useful to a wider community. For making contributions, use the ["Fork and Pull"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow to make contributions and stick as closely as possible to the following:

* Code style should comply with [PEP8](https://peps.python.org/pep-0008/) where possible. [Google's house style](https://google.github.io/styleguide/pyguide.html) is also helpful, including a good model for docstrings.
* Please use comments liberally when adding nontrivial features, and take the chance to clean up other people's code while looking at it.
* Add tests wherever possible, and use the test suite to check if you broke anything.

## Add an embedding scheme

The steps required to add a new representation scheme are:

1. Add data file to [data/element_representations](https://github.com/WMD-group/ElementEmbeddings/tree/main/src/elementembeddings/data/element_representations).
2. Edit docstring table in [core.py](https://github.com/WMD-group/ElementEmbeddings/tree/main/src/elementembeddings/core.py).
3. Edit [utils/config.py](https://github.com/WMD-group/ElementEmbeddings/tree/main/src/elementembeddings/utils/config.py) to include the representation in `DEFAULT_ELEMENT_EMBEDDINGS` and `CITATIONS`.
4. Update the documentation [reference.md](embeddings/element.md) and [README.md](https://github.com/WMD-group/ElementEmbeddings/tree/main/src/elementembeddings/data/element_representations/README.md).

## Workflow

We follow the [GitHub flow]
(<https://guides.github.com/introduction/flow/index.html>), using
branches for new work and pull requests for verifying the work.

The steps for a new piece of work can be summarised as follows:

1. Push up or create [an issue](https://guides.github.com/features/issues).
2. Create a branch from main, with a sensible name that relates to the issue.
3. Do the work and commit changes to the branch. Push the branch
   regularly to GitHub to make sure no work is accidentally lost.
4. Write or update unit tests for the code you work on.
5. When you are finished with the work, ensure that all of the unit
   tests pass on your own machine.
6. Open a pull request [on the pull request page](https://github.com/WMD-group/ElementEmbeddings/pulls).
7. If nobody acknowledges your pull request promptly, feel free to poke one of the main developers into action.

## Pull requests

For a general overview of using pull requests on GitHub look [in the GitHub docs](https://help.github.com/en/articles/about-pull-requests).

When creating a pull request you should:

* Ensure that the title succinctly describes the changes so it is easy to read on the overview page
* Reference the issue which the pull request is closing

Recommended reading: [How to Write the Perfect Pull Request](https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/)

## Dev requirements

When developing locally, it is recommended to install the python packages in `requirements-dev.txt`.

```bash
pip install -r requirements-dev.txt
```

This will allow you to run the tests locally with pytest as described in the main README,
as well as run pre-commit hooks to automatically format python files with isort and black.
To install the pre-commit hooks (only needs to be done once):

```bash
pre-commit install
pre-commit run --all-files # optionally run hooks on all files
```

Pre-commit hooks will check all files when you commit changes, automatically fixing any files which are not formatted correctly. Those files will need to be staged again before re-attempting the commit.

## Bug reports, feature requests and questions

Please use the [Issue Tracker](https://github.com/WMD-group/ElementEmbeddings/issues) to report bugs or request features in the first instance. Contributions are always welcome.
