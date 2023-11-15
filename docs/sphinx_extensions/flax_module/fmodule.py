# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sphinx directive for visualizing Flax modules.

Use directive as follows:

.. flax_module::
  :module: flax.linen
  :class: Dense

"""
import inspect
import importlib

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList

import sphinx
from sphinx.util.docutils import SphinxDirective

import sphinx.ext.autosummary.generate as ag

from flax.linen import Module


def render_module(modname: str, qualname: str, app):
    parent = importlib.import_module(modname)
    obj = getattr(parent, qualname)
    is_flax_module = False
    if type(obj) == type:
        if issubclass(obj, Module):
            is_flax_module = True
    template = ag.AutosummaryRenderer(app)
    if is_flax_module:
        template_name = "flax_module"

        # Since sphinx recognizes class attributes that are
        # functions as methods instead of the attributes they
        # are in this case, if they are in the constructor
        # signature we just set them to None so they are picked
        # up as attributes.
        sig = inspect.signature(obj)
        for name, typ in sig.parameters.items():
            if inspect.isfunction(getattr(obj, name, None)):
                # unset it
                if hasattr(obj, name):
                    try:
                        delattr(obj, name)
                    except AttributeError:
                        pass
    else:
        template_name = "flax_function_wrap"

    imported_members = False
    recursive = False
    context = {}
    return ag.generate_autosummary_content(
        qualname,
        obj,
        parent,
        template,
        template_name,
        imported_members,
        app,
        recursive,
        context,
        modname,
        qualname,
    )


class FlaxModuleDirective(SphinxDirective):
    has_content = True
    option_spec = {
        "module": directives.unchanged,
        "class": directives.unchanged,
    }

    def run(self):
        module_template = render_module(
            self.options["module"], self.options["class"], self.env.app
        )
        module_template = module_template.splitlines()

        # Create a container for the rendered nodes
        container_node = nodes.container()
        self.content = ViewList(module_template, self.content.parent)
        self.state.nested_parse(self.content, self.content_offset, container_node)

        return [container_node]


def setup(app):
    app.add_directive("flax_module", FlaxModuleDirective)

    return {
        "version": sphinx.__display_version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
