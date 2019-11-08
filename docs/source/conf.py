# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import recommonmark
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify


# -- Project information -----------------------------------------------------

project = 'CS207 final project'
copyright = '2019, Group 30'
author = 'Group 30'

# The full version, including alpha/beta/rc tags
release = '0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'recommonmark',
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.todo',
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.ifconfig',
    # 'sphinx.ext.viewcode',
    # 'sphinx.ext.githubpages',
    # 'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.jsmath',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Allowed suffixes for source files
# source_suffix = [
#     '.rst', '.md']
source_suffic = {
    '.md': 'markdown',
    '.rst': 'restructuredtext',
}

# Master document
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'manni'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = ['_themes']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

def setup(app):
    app.add_config_value('recommonmark_config', {
        #'url_resolver': lambda url: github_doc_root + url,
        'auto_toc_tree_section': 'Contents',
        'enable_eval_rst': True,
        'enable_inline_math': True,
        # 'enable_auto_doc_ref': True,
    }, True)
    app.add_transform(AutoStructify)

latex_engine = 'lualatex'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    'preamble': r'\renewcommand{\familydefault}{\sfdefault}',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    'fontpkg': r'''
        \setsansfont{Source Sans Pro}
        \setmonofont[Scale=0.9]{Consolas}
    ''' ,
    'babel': r'\usepackage[american]{babel}'
    # 'preamble':r'''
    # \usepackage{babel}
    # '''
    
    
}