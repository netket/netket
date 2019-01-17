import netket
import format
import inspect
import sys
import os
import pkgutil

default_submodules = [
    'graph', 'sampler', 'hilbert', 'operator', 'variational', 'exact'
]

def import_from_string(name):
    m = __import__(name)
    for n in name.split(".")[1:]:
        m = getattr(m, n)
    return m

def build_docs(output_directory='./', submodules=None):
    if not submodules:
        submodules = default_submodules
    if not os.path.exists(output_directory):
        os.mkdir(submod)
    for submod in submodules:
        sm = import_from_string('netket.' + submod)
        clsmembers = inspect.getmembers(sm, inspect.isclass)
        if not os.path.exists(submod):
            os.mkdir(submod)
        for clsm in clsmembers:
            print('Docs for ', clsm[1])
            markdown = format.format_class(clsm[1])
            with open(output_directory + "/" + submod + "/" + clsm[0] + ".md",
                      "w") as text_file:
                text_file.write(markdown)
