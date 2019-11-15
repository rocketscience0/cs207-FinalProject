# How to edit documentation

We're using [Sphinx](https://www.sphinx-doc.org/en/master/) to document `autodiff`. This page has instructions on how to add additional pages of documentation.

The file `index.rst` is the source for the main landing page. While it's in restructured text format, most of the documentation can be written in markdown (this page was written in markdown). To add a new page, all we need to do is create a new markdown file and tell `index.rst` to include it in the main table of contents:
1. Create a new file in the `source` directory.
1. Add that file's relative path to the `toctree` in `index.rst`
    ```rst
    .. toctree::
        :maxdepth: 2
        :caption: Contents:

        introduction.md
        test-page/how-to-edit-documentation.md
        test-page/new-test-page.md
        api-doc/modules.rst
    ```
1. Once that's done, run `make html` from the `docs` directory. Doing so will create `/docs/source/html/`. Opening `index.html` in a browser will show a local copy of our documentation page. [Our documentation site](https://cs207-autodiff.readthedocs.io/en/latest/index.html) does the same thing every time we push code to github.

The markdown syntax Sphinx uses is slightly different than Jupyter's. [This test page](test.md) has some examples of markdown that works.
