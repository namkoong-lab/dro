# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0, os.path.abspath('../dro/src'))  
print(sys.path)

project = 'dro'
copyright = '2025, Jiashuo Liu, Tianyu Wang, Peng Cui, Hongseok Namkoong, Jose Blanchet'
author = 'Jiashuo Liu, Tianyu Wang, Peng Cui, Hongseok Namkoong, Jose Blanchet'
release = '0.1.1'

# html_static_path = ['../docs']
html_baseurl = 'https://namkoong-lab.github.io/dro/' 


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax',
    "myst_parser",
    'nbsphinx',
    'sphinx_copybutton' 
]

autodoc_type_aliases = {
    'NDArray': 'numpy.ndarray',
    "Expression": "cvxpy.expressions.expression.Expression",
    'Tensor': 'torch.Tensor',
    # 'Module': 'torch.nn.Module'
}

autodoc_default_options = {
    "no-autoparams": True
}

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'python': ('https://docs.python.org/3', None)
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_typehints = "description" 

# html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
    # 'navigation_depth': 4,
    # 'collapse_navigation': False
# }


html_theme = 'piccolo_theme'
html_theme_options = {
    "source_url": 'https://github.com/namkoong-lab/dro',
    "source_icon": "github",
    "globaltoc_collapse": False,
    "banner_text": 'A GIFT to the whole DRO community!',
    "banner_hiding": "temporal",
    # "canonical_url": "", 
    # "analytics_id": "",
}


autodoc_default_options = {
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
    "private-members": False 
}

myst_enable_extensions = [
    "dollarmath",   
    "colon_fence",  
    "html_image",   
    "linkify",      
]

nbsphinx_execute = 'auto'
nbsphinx_kernel_name = 'python3'
nbsphinx_timeout = 600
nbsphinx_prompt_width = "0"
nbsphinx_include_pattern = [] 


autoclass_content = 'both'  

# add_module_names = False

nitpicky = True
nitpick_ignore = [
    ("py:class", "torch.device"),
    ("py:class", "torch.nn.Module"),
    ("py:class", "nn.Module"),
    ("py:class", "Module"),
    ("py:exc", "LinAlgError"),
    ('py:exc', 'MOTDROError'),
    ('py:exc', 'KLDROError'),
    ('py:exc', 'Chi2DROError'),
    ('py:exc', 'BayesianDROError'),
    ('py:exc', 'MMDDROError'),
    ('py:exc', 'KLDROError'),
    ('py:exc', 'ConditionalCVaRDROError'),
    ('py:exc', 'HRDROError'),
    ('py:exc', 'MarginalCVaRDROError'),
    ('py:exc', 'ORWDROError'),
    ('py:exc', 'SinkhornDROError'),
    ('py:exc', 'TVDROError'),
    ('py:exc', 'WassersteinDROError'),
    ('py:exc', 'DROError'),
    ('py:exc', 'CVaRDROError'),
    ('py:exc', 'LinearModel')
]