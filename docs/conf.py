# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0, os.path.abspath('../dro/src'))  

project = 'dro'
copyright = '2025, Jiashuo Liu, Tianyu Wang'
author = 'Jiashuo Liu, Tianyu Wang'
release = '0.1.1'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints'
]

autodoc_type_aliases = {
    'NDArray': 'numpy.ndarray',
    'Tensor': 'torch.Tensor'
}

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'python': ('https://docs.python.org/3', None)
}

autodoc_typehints = "both"

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False
}

autodoc_default_options = {
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True
}


autoclass_content = 'both'  