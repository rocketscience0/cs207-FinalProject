## Software Organization

### Directory structure

```
.
├── README.md
├── .travis.yml
├── autodiff-env.yml
├── .gitignore
└── docs
    ├── pandoc-minted.py
    └── source
	    ├── sphinx-requirements.txt
	    ├── index.rst
	    └── api-doc
	    	├── autodiff.rst
	    	└── modules.rst
└── autodiff
    ├── __init__.py
    ├── operations.py
    └── structures.py
└── tests
    ├── newtons_method.py
    └── test_operations.py
```

#### Modules

There are two modules. The `autodiff` module implements the forward mode of automatic differentiation. It contains `structures.py`, the definition of the `Number` class, and `operations.py`, the implementations of the various methods of `Number` (elementary operations, the derivatives of elementary operations). The `tests` module runs tests for `autodiff`. See below for details about testing. 

#### Testing
All tests live in `tests/test_autodiff.py`. We will use both `TravisCI` and `CodeCov` to distribute reports.