# -- Set env variables to correctly detect sphinx in NetKet
import os
import sys
import pathlib

os.environ["NETKET_SPHINX_BUILD"] = "1"
import netket as nk

# add the folder with sphinx extensions
sys.path.append(str(pathlib.PosixPath(os.getcwd()) / "sphinx_extensions"))

# -- Project information -----------------------------------------------------

project = "NetKet"
copyright = "2019-2021, The Netket authors - All rights reserved"

# The full version, including alpha/beta/rc tags
release = nk.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # "myst_parser",
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.graphviz",
    "custom_inheritance_diagram.inheritance_diagram",  # this is a custom patched version because of bug sphinx#2484
    "flax_module.fmodule",
]

# inheritance_graph_attrs = dict(rankdir="TB", size='""')
# graphviz_output_format = 'svg'

# Napoleon settings
autodoc_docstring_signature = True
autodoc_inherit_docstrings = True
allow_inherited = True
autosummary_generate = True
napoleon_preprocess_types = True

autodoc_mock_imports = ["openfermion", "qutip", "pyscf"]

# PEP 526 annotations
napoleon_attr_annotations = True

master_doc = "index"

autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_typehints = "description"

# Add any paths that contain templates here, relative to this directory.
templates_path = [
    "assets/templates",
    "assets/templates/autosummary",
    "assets/templates/sections",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
# source_suffix = {
#    ".rst": "restructuredtext",
#    ".ipynb": "myst-nb'",
#    ".md": "markdown",
#    '.myst': 'myst-nb',
# }
source_suffix = [".rst", ".ipynb", ".md"]

# Markdown parser latex support
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "html_admonition"]
myst_update_mathjax = False
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

myst_heading_anchors = 2
autosectionlabel_maxdepth = 1

main_website_base_url = "https://www.netket.org"

# -- Pre-process -------------------------------------------------
autodoc_mock_imports = ["openfermion", "qutip"]

# -- Options for HTML output -------------------------------------------------

# html_theme = "pydata_sphinx_theme"
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["assets/static"]
html_css_files = ["css/custom.css", "css/navbar.css"]  # , "css/api.css"]
html_favicon = "assets/static/favicon.ico"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "igraph": ("https://igraph.org/python/api/latest", None),
    "qutip": ("https://qutip.org/docs/latest/", None),
    "pyscf": ("https://pyscf.org/", None),
}

# (Optional) Logo. Should be small enough to fit the navbar (ideally 24x24).
# Path should be relative to the ``_static`` files directory.
html_title = "NetKet"
html_logo = "assets/static/logo_transparent.png"

html_theme_options = {
    "home_page_in_toc": False,
    "show_navbar_depth": 1,
    "show_toc_level": 3,
    "repository_url": "https://github.com/netket/netket",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs",
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "navigation_with_keys": True,
}

html_context = {
    "navbar_title": "NetKet",
    "navbar_logo": "logonav.png",
    "navbar_fixed_top": True,
    "navbar_link": (f"{main_website_base_url}", True),
    "navbar_class": "navbar",
    "navbar_links": [
        ("Posts", f"{main_website_base_url}/posts/", True),
        (
            "Get Involved",
            f"{main_website_base_url}/get_involved/",
            True,
        ),
        ("Citing", f"{main_website_base_url}/cite/", True),
        ("Documentation", "index"),
        ("API Reference", "api/api"),
    ],
    "navbar_links_right": [
        (
            '<i class="fab fa-github" aria-hidden="true"></i>',
            "https://github.com/netket/netket",
            True,
        ),
        (
            '<i class="fab fa-twitter" aria-hidden="true"></i>',
            "https://twitter.com/NetKetOrg",
            True,
        ),
        (
            '<i class="fab fa-slack" aria-hidden="true"></i>',
            "https://join.slack.com/t/mlquantum/shared_invite/zt-19wibmfdv-LLRI6i43wrLev6oQX0OfOw",
            True,
        ),
    ],
    "navbar_download_button": (
        "Get Started",
        f"{main_website_base_url}/get_started/",
        True,
    ),
}

# -- Options for myst ----------------------------------------------
nb_execution_mode = "off"
nb_execution_allow_errors = False


# do not show __init__ if it does not have a docstring
def autodoc_skip_member(app, what, name, obj, skip, options):
    # Ref: https://stackoverflow.com/a/21449475/
    exclusions = (
        "__weakref__",  # special-members
        "__doc__",
        "__module__",
        "__dict__",  # undoc-members
        "__new__",
    )
    exclude = name in exclusions
    if name == "__init__":
        exclude = True if obj.__doc__ is None else False
    return True if (skip or exclude) else None


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    # app.connect('autodoc-process-docstring', warn_undocumented_members);

    # fix modules
    # process_module_names(netket)
    # process_module_names(netket.experimental)


import netket
import netket.experimental
import inspect


def process_module_names(module, modname="", inner=0):
    """
    This function goes through everything that is exported through __all__ in every
    module, recursively, and if it hits classes or functions it changes their __module__
    so that it reflects the one we want printed in the docs (instead of the actual one).

    This fixes the fact that for example netket.graph.Lattice is actually
    netket.graph.lattice.Lattice
    """
    if hasattr(module, "__all__"):
        for subm in module.__all__:
            obj = getattr(module, subm)
            process_module_names(obj, f"{module.__name__}", inner=inner + 1)
    elif inspect.isclass(module):
        module.__module__ = modname
    elif inspect.isfunction(module):
        module.__module__ = modname
