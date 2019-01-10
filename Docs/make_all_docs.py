import netket
import format
import inspect
import sys
import os
import pkgutil


def import_from_string(name):
    m = __import__(name)
    for n in name.split(".")[1:]:
        m = getattr(m, n)
    return m


submodules = ['graph', 'sampler', 'hilbert', 'operator']

for submod in submodules:
    sm = import_from_string('netket.' + submod)
    clsmembers = inspect.getmembers(sm, inspect.isclass)
    if not os.path.exists(submod):
        os.mkdir(submod)
    for clsm in clsmembers:
        print('Docs for ', clsm[1])
        markdown = format.format_class(clsm[1])
        with open(submod + "/" + clsm[0] + ".md", "w") as text_file:
            text_file.write(markdown)
