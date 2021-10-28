# EMACS settings: -*- coding: utf-8; tab-width: 2; indent-tabs-mode: t -*-
# vim: tabstop=2:shiftwidth=2:noexpandtab
# kate: tab-width 2; replace-tabs off; indent-width 2;
# =============================================================================
# Authors:            Patrick Lehmann
#
# Package installer:  A modified version of sphinx.ext.inheritance_diagram
#
#
# License:
# ============================================================================
# Copyright 2017-2019 Patrick Lehmann - Bötzingen, Germany
# Copyright (c) 2007-2019 by the Sphinx team (see AUTHORS file).
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-License-Identifier: BSD-2-Clause
# ============================================================================
#
import setuptools

projectName =           "sphinx.inheritance_diagram"
projectNameWithPrefix = "btd." + projectName
version =               "2.3.1.post1"

setuptools.setup(
	name=projectNameWithPrefix,
	version=version,

	author="Sphinx team, Patrick Lehmann",
	author_email="Paebbels@gmail.com",
	# maintainer="Patrick Lehmann",
	# maintainer_email="Paebbels@gmail.com",

	description="Embedding diagrams rendered with inheritance_diagram.",
	long_description_content_type="text/markdown",

	# download_url="https://github.com/buildthedocs/sphinx.inheritance_diagram/tarball/0.1.0",

	packages=setuptools.find_namespace_packages(),
	classifiers=[
		"License :: OSI Approved :: BSD License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3 :: Only",
		"Programming Language :: Python :: 3.4",
		"Programming Language :: Python :: 3.5",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Development Status :: 5 - Production/Stable",
		"Intended Audience :: Developers",
		"Topic :: Documentation :: Sphinx",
		"Framework :: Sphinx :: Extension"
	],
	keywords="Sphinx Documentation Inheritance Diagram Graphviz",

	python_requires='>=3.4',
	install_requires=[],
	# provides=
	# obsoletes=
)
