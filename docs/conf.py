# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))

project = 'sam-ml-py'
copyright = f"2022 - {datetime.now().year}, Samuel Brinkmann (MIT License)"
author = 'Samuel Brinkmann'
import sam_ml

release = sam_ml.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sphinx_copybutton
import sphinx_rtd_theme
from sphinx.application import Sphinx

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Specify how to identify the prompt when copying code snippets
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_exclude = "style"

# If True, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "sam_ml"

# If false, no module index is generated.
html_domain_indices = False

# If false, no index is generated.
html_use_index = False
