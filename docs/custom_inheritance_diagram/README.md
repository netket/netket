[![Sourcecode on GitHub](https://img.shields.io/badge/BuildTheDocs-sphinx.inheritance_diagram-323131.svg?logo=github&longCache=true)](https://github.com/buildthedocs/sphinx.autoprogram)
<!--[![License](https://img.shields.io/badge/Apache%20License,%202.0-bd0000.svg?longCache=true&label=code%20license&logo=Apache&logoColor=D22128)](LICENSE.md)-->
[![GitHub tag (latest SemVer incl. pre-release)](https://img.shields.io/github/v/tag/buildthedocs/sphinx.inheritance_diagram?logo=GitHub&include_prereleases)](https://github.com/buildthedocs/sphinx.inheritance_diagram/tags)
[![GitHub release (latest SemVer incl. including pre-releases)](https://img.shields.io/github/v/release/buildthedocs/sphinx.inheritance_diagram?logo=GitHub&include_prereleases)](https://github.com/buildthedocs/sphinx.inheritance_diagram/releases/latest)
[![GitHub release date](https://img.shields.io/github/release-date/buildthedocs/sphinx.inheritance_diagram?logo=GitHub&)](https://github.com/buildthedocs/sphinx.inheritance_diagram/releases)
[![Libraries.io status for latest release](https://img.shields.io/librariesio/release/pypi/btd.sphinx.inheritance_diagram)](https://libraries.io/github/buildthedocs/sphinx.inheritance_diagram)
[![Requires.io](https://img.shields.io/requires/github/buildthedocs/sphinx.inheritance_diagram)](https://requires.io/github/buildthedocs/sphinx.inheritance_diagram/requirements/?branch=master)  
<!--[![Travis](https://img.shields.io/travis/com/buildthedocs/sphinx.inheritance_diagram?logo=Travis)](https://travis-ci.com/buildthedocs/sphinx.inheritance_diagram)-->
[![PyPI](https://img.shields.io/pypi/v/btd.sphinx.inheritance_diagram?logo=PyPI)](https://pypi.org/project/sphinx.inheritance_diagram/)
![PyPI - Status](https://img.shields.io/pypi/status/btd.sphinx.inheritance_diagram?logo=PyPI)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/btd.sphinx.inheritance_diagram?logo=PyPI)
[![Dependent repos (via libraries.io)](https://img.shields.io/librariesio/dependent-repos/pypi/btd.sphinx.inheritance_diagram)](https://github.com/buildthedocs/sphinx.inheritance_diagram/network/dependents)  
<!-- [![Read the Docs](https://img.shields.io/readthedocs/btd-sphinx-inheritance_diagram)](https://btd-sphinx-inheritance_diagram.readthedocs.io/en/latest/)-->

# btd.sphinx.inheritance_diagram

This is a patched version of [`sphinx.ext.inheritance_diagram`](https://github.com/sphinx-doc/sphinx).

This package is required, if inheritance diagrams (e.g. `sphinx.ext.inheritance_diagram`)
are enabled in Sphinx **AND** Sphinx is used with our patched variation of
`sphinx.ext.graphviz` called [`btd.sphinx.graphviz`](https://github.com/buildthedocs/sphinx.graphviz).
In this example, `inheritance_diagram` has an internal cross reference to
`graphviz`, which cannot be satisfied, because the graphviz extension has been
exchanged.


> **Note:**  
> Patched versions of internal packages from Sphinx are released as M.M.P.postN
> versions. So `2.3.1.post1` is the patched module version derived from Sphinx
> `2.3.1`.

--------------------

## Added features

* `2.3.1.post1`
  * Changed dependencies `sphinx.ext.graphviz` to `btd.sphinx.graphviz`.

--------------------

## Install using `pip`

```
$ pip install btd.sphinx.inheritance_diagram
```

----------------------

SPDX-License-Identifier: BSD-2-Clause
