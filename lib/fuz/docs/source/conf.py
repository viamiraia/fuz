# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from dataclasses import asdict

from sphinxawesome_theme import ThemeOptions
from sphinxawesome_theme.postprocess import Icons

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fuz'
copyright = '2024, miraia s chiou'
author = 'miraia s chiou'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinxawesome.deprecated',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'
html_static_path = ['_static']


# Select theme for both light and dark mode
pygments_style = 'colorful'
# Select a different theme for dark mode
pygments_style_dark = 'lightbulb'


html_permalinks_icon = Icons.permalinks_icon


theme_options = ThemeOptions(
    # Add your theme options. For example:
    show_breadcrumbs=True,
    main_nav_links={'About': '/about'},
)

html_theme_options = asdict(theme_options)
