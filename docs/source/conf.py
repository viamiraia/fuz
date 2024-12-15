# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# %% Imports and Setup
import datetime as dt
import sys
from dataclasses import asdict
from pathlib import Path

import fuz

# Add module path to sys.path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path.absolute()))

current_date = dt.datetime.now(tz=dt.timezone.utc)

# %% General Configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fuz'
year_string = f'2024-{current_date.year}' if current_date.year != 2024 else '2024'
copyright = f'{year_string}, miraia s chiou'
author = 'miraia s chiou'
release = fuz.__version__

# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'sphinx_copybutton',
    'autoapi.extension',
    'myst_parser',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = ['Thumbs.db', '.DS_Store']
language = 'en'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# %% Extension Configuration

intersphinx_mapping = {'sphinx': ('https://www.sphinx-doc.org/en/master', None)}
autosummary_generate = True
autodoc_typehints = 'description'

# https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_exclude = '.linenos, .gp, .go'

# %%% MyST Config
myst_enable_extensions = [
    'strikethrough',
    'amsmath',
    'dollarmath',
    'linkify',
    'substitution',
    'colon_fence',
    'deflist',
    'tasklist',
    'fieldlist',
    'attrs_inline',
    'html_admonition',
    'html_image',
]
myst_heading_anchors = 3


# %%% Autodoc Config
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'ignore-module-all': True,
    'member-order': 'groupwise',
}

typehints_defaults = 'braces'
always_use_bars_union = True

# %%% Napoleon settings
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True

# %%% AutoAPI
# https://sphinx-autoapi.readthedocs.io/en/latest/index.html
autoapi_dirs = [str(root_path / 'fuz')]
autoapi_root = 'api'
autoapi_keep_files = True
autoapi_member_order = 'groupwise'
autoapi_ignore = ['*/demo/*']
# autoapi_add_toctree_entry = False

# %% Theme and HTML Configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_title = 'fuz docs'
html_last_updated_fmt = ''
html_sourcelink_suffix = ''
# TODO(viamiraia): make and add icons
# html_logo = "_static/logo.svg"
# html_favicon = "_static/logo.svg"

html_theme_options = {
    'external_links': [
        {
            'url': 'https://code.raia.fun',
            'name': "raia's code",
        },
    ],
    'icon_links': [
        {
            'name': 'bluesky',
            'url': 'https://bsky.app/profile/raia.fun',
            'icon': 'fa-brands fa-bluesky',
        },
        {
            'name': 'ko-fi',
            'url': 'https://ko-fi.com/viamiraia',
            'icon': 'fa-solid fa-hand-holding-dollar',
        },
        {
            'name': 'pypi',
            'url': 'https://pypi.org/project/fuz',
            'icon': 'fa-solid fa-cubes',
        },
        {
            'name': 'github',
            'url': 'https://github.com/viamiraia/fuz',
            'icon': 'fa-brands fa-github',
        },
    ],
    'use_edit_page_button': True,
    'show_toc_level': 3,
    'show_version_warning_banner': True,
    'footer_start': ['sphinx-version'],
    'footer_center': ['copyright'],
    'secondary_sidebar_items': ['page-toc', 'edit-this-page', 'sourcelink'],
    'switcher': {
        'json_url': 'https://code.raia.fun/fuz/en/latest/_static/switcher.json',
        'version_match': release,
    },
    'show_nav_level': 3,
}
html_context = {
    'github_user': 'viamiraia',
    'github_repo': 'fuz',
    'github_version': 'main',
    'doc_path': 'docs',
    'default_mode': 'light',
}
#
