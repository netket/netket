import contextlib
import os
import re
import shlex
import subprocess
import sys

from distutils import log
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

# NOTE(twesterhout): distutils and setuptools use a pretty strange way of
# handling command line arguments, where some arguments are propagated to
# sub-commands, some are not... All in all, believe me, it's really hacky :)
# The simplest way I found to reliably pass some cmake arguments to build_ext
# command is to use a hack of our own. We hijack the sys.argv and manually
# extract all the cmake-related options. We then store them in a module-local
# variable and use for building of extensions based on CMake.

# Poor man's command-line options parsing
def steal_cmake_args(args):
    _ARG_PREFIX = '--cmake-args='
    def _unquote(x):
        m = re.match(r"'(.*)'", x)
        if m:
            return m.group(1)
        m = re.match(r'"(.*)"', x)
        if m:
            return m.group(1)
        return x
    stolen_args = [x for x in args if x.startswith(_ARG_PREFIX)]
    for x in stolen_args:
        args.remove(x)

    if len(stolen_args) > 0:
        cmake_args = sum((shlex.split(_unquote(x[len(_ARG_PREFIX):])) for x in stolen_args), [])
    else:
        try:
            cmake_args = shlex.split(os.environ['NETKET_CMAKE_FLAGS'])
        except KeyError:
            cmake_args = []
    return cmake_args

_CMAKE_ARGS = steal_cmake_args(sys.argv)

class CMakeExtension(Extension):
    """
    :param sourcedir: Specifies the directory (if a relative path is used, it's
                      relative to this "setup.py" file) where the "CMakeLists.txt"
                      for building the extension `name` can be found.
    """
    def __init__(self, name, sourcedir='.'):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.join(os.path.realpath('.'), sourcedir)

class build_ext(build_ext_orig, object):
    # The only function we override. We use custom logic for building
    # CMakeExtensions, but leave everything else as is.
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            cwd = os.getcwd()
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            lib_dir = os.path.abspath(self.build_lib)
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            cmake_args = _CMAKE_ARGS
            cmake_args.append(
                '-DNETKET_PYTHON_VERSION={}.{}.{}'.format(*sys.version_info[:3]))
            cmake_args.append(
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(lib_dir))

            os.chdir(self.build_temp)
            try:
                output = subprocess.check_output(
                    ['cmake', ext.sourcedir] + cmake_args, stderr=subprocess.STDOUT)
                if self.distribution.verbose:
                    log.info(output.decode())
                if not self.distribution.dry_run:
                    output = subprocess.check_output(
                        ['cmake', '--build', '.'], stderr=subprocess.STDOUT)
                    if self.distribution.verbose:
                        log.info(output.decode())
            except subprocess.CalledProcessError as e:
                if hasattr(ext, 'optional'):
                    if not ext.optional:
                        self.warn(e.output.decode())
                        raise
                    self.warn('building extension "{}" failed:\n{}'.format(ext.name, e.output.decode()))
                else:
                    self.warn(e.output.decode())
                    raise
            os.chdir(cwd)
        else:
            if sys.version_info >= (3, 0):
                super().initialize_options()
            else:
                super(build_ext_orig, self).initialize_options()

setup(
    name='netket',
    version='2.0',
    author='Giuseppe Carleo et al.',
    description='NetKet',
    url='http://github.com/netket/netket',
    author_email='netket@netket.org',
    license='Apache',
    ext_modules=[CMakeExtension('netket')],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    install_requires=[
        'mpi4py'
    ]
)
