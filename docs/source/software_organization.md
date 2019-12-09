## Software Organization

### Directory structure

```
.
├── README.md
├── .travis.yml
├── autodiff-env.yml
├── .gitignore
└── docs
    ├── bfgs.ipynb
    ├── default-header.py
    ├── make.bat
    ├── Makefile
    ├── milestone1.ipynb
    ├── milestone1.md
    ├── milestone1.pdf
    ├── milestone2.pdf
    ├── optimization_example.ipynb
    ├── pandoc-minted.py
    ├── root_finding_test.ipynb
    ├── toipynb.sh
    ├── topdf.sh
    └── source
	    ├── conf.py
	    ├── How_to_install.md
	    ├── How_to_use.md
	    ├── implementation.md
	    ├── index.md
	    ├── introduction.md
	    ├── proposed-extension.md
	    ├── root_finding_test.md
	    ├── software_organization.md
	    ├── sphinx-requirements.txt
	    ├── index.rst
	    └── api-doc
	    	├── autodiff.rst
	    	└── modules.rst
	    └── examples
	    	├── bfgs.md
	    	├── index.rst
	    	└── root_finding.md
	    └── image
	    	├── equation1.svg
	    	├── equation2.svg
	    	├── milestone1_computation_graph.png
	    	├── milestone1_evaluation_table.png
	    	└── equation3.svg
	    └── test-page
	    	└── test.md
└── autodiff
    ├── __init__.py
    ├── _utlities.py
    ├── operations.py
    ├── optimizations.py
    ├── root_finding.py
    └── structures.py
└── tests
    ├── newtons_method.py
    ├── test_array.py
    ├── test_operations.py
    └── test_optimization.py
```

#### Modules

There are two modules. The `autodiff` module implements the forward mode of automatic differentiation. It contains `structures.py`, the definition of the `Number` and `Array` class, and `operations.py`, the implementations of the various elementary operations and the derivatives of elementary operations. The `root_finding.py` and `optimizations.py` that performs various root-finding and optimization methods as a proposed extension. The `tests` module runs tests for `autodiff`. See below for details about testing. 

#### Testing
All tests live in `tests/test_autodiff.py`. We will use both `TravisCI` and `CodeCov` to distribute reports.