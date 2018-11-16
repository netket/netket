import contextlib
import os
import re
import sys
import subprocess
from subprocess import CalledProcessError

import shlex

from distutils import log
from distutils.command.build import build as build_orig
from distutils.core import Command
from distutils.errors import DistutilsError

import setuptools
from setuptools import setup, Extension
from setuptools.command.install import install as install_orig
from setuptools.command.build_ext import build_ext as build_ext_orig

# Command class we'll use in `setup()`
cmdclass = {}

# Patches a few pre-installed commands to accept `--cmake-args` command line
# argument.
# NOTE(twesterhout): I'd like to do it in a loop, but Python 2's super() is too
# buggy for that...
class _CMakeBuild(build_orig, object):
    user_options = build_orig.user_options + [
        ("cmake-args=", None, "Arguments passed directly to CMake"),
    ]

    def initialize_options(self):
        self.cmake_args = None
        if sys.version_info >= (3, 0):
            super().initialize_options()
        else:
            super(_CMakeBuild, self).initialize_options()

cmdclass['build'] = _CMakeBuild

class _CMakeInstall(install_orig, object):
    user_options = install_orig.user_options + [
        ("cmake-args=", None, "Arguments passed directly to CMake"),
    ]

    def initialize_options(self):
        self.cmake_args = None
        if sys.version_info >= (3, 0):
            super().initialize_options()
        else:
            super(_CMakeInstall, self).initialize_options()

cmdclass['install'] = _CMakeInstall

# Our custom version of build_ext command that uses CMake
class CMakeBuildExt(build_ext_orig, object):
    description = "Build C/C++ extensions using CMake"

    user_options = [
        ("cmake-args=", "o", "Arguments passed directly to CMake"),
        ("build-temp=", "t", "Directory for temporary files (build by-products)"),
        # Making Python 2 happy
        ("library-dirs=", "L", "Unused"),
    ]

    def initialize_options(self):
        self.extensions = None
        self.package = None
        self.cmake_args = None
        # self.build_temp = None
        self.build_args = None
        # self.build_lib = None
        # Making Python 2 happy
        # self.library_dirs = []
        if sys.version_info >= (3, 0):
            super().initialize_options()
        else:
            super(CMakeBuildExt, self).initialize_options()
        

    def finalize_options(self):
        if sys.version_info >= (3, 0):
            super().finalize_options()
        else:
            super(CMakeBuildExt, self).finalize_options()
        # We steal cmake_args from both install and build commands. Also we
        # need to know in which directory to build the project -- build_temp
        # and where to save the compiled modules -- build_lib
        self.set_undefined_options('install', ('cmake_args', 'cmake_args'))
        self.set_undefined_options('build', ('cmake_args', 'cmake_args'),
        #                                    ('build_lib', 'build_lib'),
        #                                    ('build_temp', 'build_temp')
                                  )

        # The standard usage of --cmake-args will be with --install-option
        # which means that cmake_args will be quoted and for some reason CMake
        # doesn't play well with it.
        def unquote(x):
            m = re.match(r"'(.*)'", x)
            if m:
                return m.group(1)
            m = re.match(r'"(.*)"', x)
            if m:
                return m.group(1)
            return x

        if self.cmake_args is None:
            self.warn("No CMake options specified")
            self.cmake_args = []
        else:
            # shlex.split splits the string into words correctly handling
            # quoted strings
            self.cmake_args = shlex.split(unquote(self.cmake_args))

        # Use the exact same Python version in CMake as the current process
        self.cmake_args.append(
            '-DNETKET_PYTHON_VERSION={}.{}.{}'.format(*sys.version_info[:3]))

        self.build_args = []

    def run(self):
        for ext in self.extensions:
            with self._filter_build_errors(ext):
                self.build_extension(ext)

    @contextlib.contextmanager
    def _filter_build_errors(self, ext):
        try:
            yield
        except (DistutilsError, CalledProcessError) as e:
            self.warn('Failed: \n{}'.format(e.output.decode()))
            if not ext.optional:
                raise
            self.warn('building extension "{}" failed: {}'.format(ext.name, e))

    def build_extension(self, ext):
        cwd = os.getcwd()

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        lib_dir = os.path.abspath(self.build_lib)
        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        cmake_args = self.cmake_args
        cmake_args.append(
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(lib_dir))

        os.chdir(self.build_temp)
        output = subprocess.check_output(
            ['cmake', ext.sourcedir] + cmake_args, stderr=subprocess.STDOUT)
        if self.distribution.verbose:
            log.info(output.decode())
        if not self.distribution.dry_run:
            output = subprocess.check_output(
                ['cmake', '--build', '.'] + self.build_args, stderr=subprocess.STDOUT)
            if self.distribution.verbose:
                log.info(output.decode())
        os.chdir(cwd)

    # def get_outputs(self):
    #     outputs = []
    #     for ext in self.extensions:
    #         outputs.append(self.get_ext_fullpath(ext.name))
    #     return outputs

    # def get_source_files(self):
    #     return []

    # The rest is taken from CPython's implementation of build_ext

    # -- Name generators -----------------------------------------------
    # (extension names, filenames, whatever)
    # def get_ext_fullpath(self, ext_name):
    #     """Returns the path of the filename for a given extension.

    #     The file is located in `build_lib`.
    #     """
    #     fullname = self.get_ext_fullname(ext_name)
    #     modpath = fullname.split('.')
    #     filename = self.get_ext_filename(modpath[-1])

    #     filename = os.path.join(*modpath[:-1]+[filename])
    #     return os.path.join(self.build_lib, filename)

    # def get_ext_fullname(self, ext_name):
    #     """Returns the fullname of a given extension name.

    #     Adds the `package.` prefix"""
    #     if self.package is None:
    #         return ext_name
    #     else:
    #         return self.package + '.' + ext_name

    # def get_ext_filename(self, ext_name):
    #     r"""Convert the name of an extension (eg. "foo.bar") into the name
    #     of the file from which it will be loaded (eg. "foo/bar.so", or
    #     "foo\bar.pyd").
    #     """
    #     from distutils.sysconfig import get_config_var
    #     ext_path = ext_name.split('.')
    #     ext_suffix = get_config_var('EXT_SUFFIX')
    #     return os.path.join(*ext_path) + ext_suffix

class CMakeExtension(Extension):
    """
    :param sourcedir: Specifies the directory (if a relative path is used, it's
                      relative to this "setup.py" file) where the "CMakeLists.txt"
                      for building the extension `name` can be found.
    """
    def __init__(self, name, sourcedir='.'):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.join(os.path.realpath('.'), sourcedir)

cmdclass['build_ext'] = CMakeBuildExt

print(setuptools.__version__)

setup(
    name='netket',
    version='0.1',
    author='Giuseppe Carleo et al.',
    description='NetKet',
    url='http://github.com/netket/netket',
    author_email='netket@netket.org',
    license='Apache',
    ext_modules=[CMakeExtension('netket')],
    cmdclass=cmdclass,
    zip_safe=False,
    test_requires=[
        'pytest',
        'numpy',
        'networkx',
        'matplotlib', # TODO: Do we need it?
    ],
    install_requires=[
        'mpi4py' # TODO: remove me!
    ]
)
