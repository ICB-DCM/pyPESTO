#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# pyPESTO documentation build configuration file, created by
# sphinx-quickstart on Mon Jul 30 08:30:38 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath('../'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '3.0.4'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # include documentation from docstrings
    'sphinx.ext.autodoc',
    # generate autodoc summaries
    'sphinx.ext.autosummary',
    # use mathjax for latex formulas
    'sphinx.ext.mathjax',
    # link to code
    'sphinx.ext.viewcode',
    # link to other projects' docs
    'sphinx.ext.intersphinx',
    # support numpy and google style docstrings
    'sphinx.ext.napoleon',
    # support todo items
    'sphinx.ext.todo',
    # source parser for jupyter notebook files
    'nbsphinx',
    # code highlighting in jupyter cells
    'IPython.sphinxext.ipython_console_highlighting',
    # support markdown-based docs
    'myst_parser',
    # bibtex references
    'sphinxcontrib.bibtex',
    # ensure that jQuery is installed
    'sphinxcontrib.jquery',
]

# default autodoc options
# list for special-members seems not to be possible before 1.8
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__, __call__',
    'imported-members': True,
    'show-inheritance': True,
    'autodoc_inherit_docstrings': True,
}
autodoc_mock_imports = ["amici"]

# links for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'petab': (
        'https://petab.readthedocs.io/projects/libpetab-python/en/latest/',
        None,
    ),
    'amici': ('https://amici.readthedocs.io/en/latest/', None),
}

bibtex_bibfiles = ["using_pypesto.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'pyPESTO'
copyright = '2018, The pyPESTO developers'
author = 'The pyPESTO developers'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
import pypesto

version = pypesto.__version__
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    'example/tmp',
    'README.md',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# Add notebooks prolog to Google Colab and nbviewer
nbsphinx_prolog = r"""
{% set docname = 'github/icb-dcm/pypesto/blob/main/doc/' + env.doc2path(env.docname, base=None) %}
.. raw:: html

    <div class="note">
      <a href="https://colab.research.google.com/{{ docname|e }}" target="_blank">
      <img src="../_static/colab-badge.svg" alt="Open in Colab"/></a>
      <a href="https://nbviewer.jupyter.org/{{ docname|e }}" target="_blank">
      <img src="../_static/nbviewer-badge.svg" alt="Open in nbviewer"/></a>
    </div>

"""

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': -1,
}

# Title
html_title = "pyPESTO documentation"
# Navigation bar title
html_short_title = "pyPESTO"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Favicon
html_favicon = "logo/logo_favicon.png"

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pyPESTOdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        'pyPESTO.tex',
        'pyPESTO Documentation',
        'The pyPESTO developers',
        'manual',
    ),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'pypesto', 'pyPESTO Documentation', [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        'pyPESTO',
        'pyPESTO Documentation',
        author,
        'pyPESTO',
        'One line description of project.',
        'Miscellaneous',
    ),
]
