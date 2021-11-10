import sphinx_bootstrap_theme

# -- Project information -----------------------------------------------------

project = "netket"
copyright = "2019-2021, The Netket authors - All rights reserved"
author = "Giuseppe Carleo et al."

# The full version, including alpha/beta/rc tags
release = "v3.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_reredirects",
    "sphinx_panels",
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.graphviz",
    "btd.sphinx.inheritance_diagram",  # this is a custom patched version because of bug sphinx#2484
]

# inheritance_graph_attrs = dict(rankdir="TB", size='""')
# graphviz_output_format = 'svg'

# Napoleon settings
autodoc_docstring_signature = True
autodoc_inherit_docstrings = True
allow_inherited = True
autosummary_generate = True
napoleon_preprocess_types = True

# PEP 526 annotations
napoleon_attr_annotations = True

panels_add_bootstrap_css = False

master_doc = "index"

autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_typehints = "description"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", "_templates/autosummary"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Markdown parser latex support
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_update_mathjax = False
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# -- Options for HTML output -------------------------------------------------

html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# html_js_files = ["https://kit.fontawesome.com/7c145f31db.js"]
html_css_files = [
    "jumbo-style.css",
    "css/all.min.css",
    "css/custom.css",
    "css/rtd_theme.css",
]

html_js_files = [
    "js/rtd_theme.js",
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    # "networkx": ("https://networkx.org/doc/reference/", None),
}

# (Optional) Logo. Should be small enough to fit the navbar (ideally 24x24).
# Path should be relative to the ``_static`` files directory.
html_logo = "_static/logonav.png"

# Theme options are theme-specific and customize the look and feel of a
# theme further.
html_theme_options = {
    # Navigation bar title. (Default: ``project`` value)
    "navbar_title": "NetKet",
    # Tab name for entire site. (Default: "Site")
    "navbar_site_name": "Site",
    # A list of tuples containing pages or urls to link to.
    # Valid tuples should be in the following forms:
    #    (name, page)                 # a link to a page
    #    (name, "/aa/bb", 1)          # a link to an arbitrary relative url
    #    (name, "http://example.com", True) # arbitrary absolute url
    # Note the "1" or "True" value above as the third argument to indicate
    # an arbitrary url.
    "navbar_links": [
        ("Get Started", "getting_started"),
        ("Documentation", "docs/getting_started"),
        ("Tutorials", "tutorials"),
        ("Citing NetKet", "citing"),
        ("About", "about"),
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
    ],
    # Render the next and previous page links in navbar. (Default: true)
    "navbar_sidebarrel": False,
    # Render the current pages TOC in the navbar. (Default: true)
    "navbar_pagenav": False,
    # Tab name for the current pages TOC. (Default: "Page")
    "navbar_pagenav_name": "Page",
    # Global TOC depth for "site" navbar tab. (Default: 1)
    # Switching to -1 shows all levels.
    "globaltoc_depth": 10,
    # Include hidden TOCs in Site navbar?
    #
    # Note: If this is "false", you cannot have mixed ``:hidden:`` and
    # non-hidden ``toctree`` directives in the same page, or else the build
    # will break.
    #
    # Values: "true" (default) or "false"
    "globaltoc_includehidden": "false",
    # HTML navbar class (Default: "navbar") to attach to <div> element.
    # For black navbar, do "navbar navbar-inverse"
    "navbar_class": "navbar",
    # Fix navigation bar to top of page?
    # Values: "true" (default) or "false"
    "navbar_fixed_top": "true",
    # Location of link to source.
    # Options are "nav" (default), "footer" or anything else to exclude.
    "source_link_position": "none",
    # Bootswatch (http://bootswatch.com/) theme.
    #
    # Options are nothing (default) or the name of a valid theme
    # such as "cosmo" or "sandstone".
    #
    # The set of valid themes depend on the version of Bootstrap
    # that's used (the next config option).
    #
    # Currently, the supported themes are:
    # - Bootstrap 2: https://bootswatch.com/2
    # - Bootstrap 3: https://bootswatch.com/3
    "bootswatch_theme": "flatly",
    # Choose Bootstrap version.
    # Values: "3" (default) or "2" (in quotes)
    "bootstrap_version": "3",
}

html_sidebars = {
    "docs/*": ["custom_localtoc.html"],
    "docs/_generated/**/*": ["custom_localtoc.html"],
    "modules/*": ["custom_localtoc.html"],
}

## redirects
redirects = {
    "documentation": "docs/getting_started.html",
}


# do not show __init__ if it does not have a docstring
def autodoc_skip_member(app, what, name, obj, skip, options):
    # Ref: https://stackoverflow.com/a/21449475/
    exclusions = (
        "__weakref__",  # special-members
        "__doc__",
        "__module__",
        "__dict__",  # undoc-members
    )
    exclude = name in exclusions
    if name == "__init__":
        exclude = True if obj.__doc__ is None else False
    return True if (skip or exclude) else None


## bug in sphinx: take docstring
# def warn_undocumented_members(app, what, name, obj, options, lines):
#    if name.startswith("netket"):
#        print(f"Autodoc dostuff: {what}, {name}, {obj}, {lines}, {options}")
#        print(f"the type is {type(obj)}")
#        if obj.__doc__ == None:
#
#    else:
#        print(f"Autodoc cacca: {what}, {name}, {obj}, {lines}, {options}")


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    # app.connect('autodoc-process-docstring', warn_undocumented_members);

    # fix modules
    process_module_names(netket)
    process_module_names(netket.experimental)


import netket
import netket.experimental
import inspect


def process_module_names(module, modname="", inner=0):
    """
    This function goes through everything that is exported through __all__ in every
    module, recursively, and if it hits classes or functions it chagnes their __module__
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
