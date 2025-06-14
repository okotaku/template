"""Sphinx configuration for the template project."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------

project = "Template"
copyright = "2024, Template Contributors"
author = "Template Contributors"

# Get version from pyproject.toml
try:
    from importlib.metadata import version
    release = version("modules")
    version = ".".join(release.split(".")[:2])  # X.Y
except Exception:
    version = "0.1.0"
    release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    # Sphinx built-in extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",

    # Third-party extensions
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "autoapi.extension",
]

# MyST Parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Auto-API configuration
autoapi_type = "python"
autoapi_dirs = ["../modules"]
autoapi_add_toctree_entry = False
autoapi_generate_api_docs = True
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon configuration (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# External links
extlinks = {
    "issue": ("https://github.com/okotaku/template/issues/%s", "issue %s"),
    "pr": ("https://github.com/okotaku/template/pull/%s", "PR %s"),
}

# Source file configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = f"{project} {version}"

html_theme_options = {
    "source_repository": "https://github.com/okotaku/template/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/okotaku/template",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#336790",
        "color-brand-content": "#336790",
    },
    "dark_css_variables": {
        "color-brand-primary": "#E0D7FF",
        "color-brand-content": "#E0D7FF",
    },
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "",
    "fontpkg": "",
}

latex_documents = [
    (master_doc, "template.tex", f"{project} Documentation", author, "manual"),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, "template", f"{project} Documentation", [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "template",
        f"{project} Documentation",
        author,
        "template",
        "A modern Python project template.",
        "Miscellaneous",
    ),
]

