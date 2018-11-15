import contextlib
import os
import re
import sys
import subprocess
from subprocess import CalledProcessError

import shlex
import pathlib

from distutils import log
from distutils.command.build import build as build_orig
from distutils.core import Command
from distutils.errors import DistutilsError

from setuptools import setup, Extension
from setuptools.command.install import install as install_orig

# Command class we'll use in `setup()`
cmdclass = {}

# Patches a few pre-installed commands to accept `--cmake-args` command line
# argument.
for _Command in [build_orig, install_orig]:
    class _CMakeCommand(_Command):
        user_options = _Command.user_options + [
            ("cmake-args=", None, "Arguments passed directly to CMake"),
        ]

        def initialize_options(self):
            self.cmake_args = None
            super().initialize_options()

    cmdclass[_Command.__name__] = _CMakeCommand

# Our custom version of build_ext command that uses CMake
class CMakeBuildExt(Command):
    description = "Build C/C++ extensions using CMake"

    user_options = [
        ("cmake-args=", "o", "Arguments passed directly to CMake"),
        ("build-temp=", "t", "Directory for temporary files (build by-products)"),
    ]

    def initialize_options(self):
        self.extensions = None
        self.package = None
        self.cmake_args = None
        self.build_temp = None
        self.build_args = []
        self.build_lib = None

    def finalize_options(self):
        # We steal cmake_args from both install and build commands. Also we
        # need to know in which directory to build the project -- build_temp
        # and where to save the compiled modules -- build_lib
        self.set_undefined_options('install', ('cmake_args', 'cmake_args'))
        self.set_undefined_options('build', ('cmake_args', 'cmake_args'),
                                            ('build_lib', 'build_lib'),
                                            ('build_temp', 'build_temp'))
        if self.cmake_args:
            self.warn("cmake_args changed in build_ext: {}".format(self.cmake_args))

        if self.package is None:
            self.package = self.distribution.ext_package

        self.extensions = self.distribution.ext_modules

        if self.build_temp is None:
            self.build_temp = "build"

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
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        lib_dir = pathlib.Path(self.build_lib).absolute()
        lib_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = self.cmake_args
        cmake_args.append(
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(str(lib_dir)))

        os.chdir(str(build_temp))
        output = subprocess.check_output(
            ['cmake', ext.sourcedir] + cmake_args, stderr=subprocess.STDOUT)
        if self.distribution.verbose:
            log.info(output.decode())
        if not self.distribution.dry_run:
            output = subprocess.check_output(
                ['cmake', '--build', '.'] + self.build_args, stderr=subprocess.STDOUT)
            if self.distribution.verbose:
                log.info(output.decode())
        os.chdir(str(cwd))

    def get_outputs(self):
        outputs = []
        for ext in self.extensions:
            outputs.append(self.get_ext_fullpath(ext.name))
        log.info(outputs)
        return outputs

    def get_source_files(self):
        return []

    # The rest is taken from CPython's implementation of build_ext

    # -- Name generators -----------------------------------------------
    # (extension names, filenames, whatever)
    def get_ext_fullpath(self, ext_name):
        """Returns the path of the filename for a given extension.

        The file is located in `build_lib`.
        """
        fullname = self.get_ext_fullname(ext_name)
        modpath = fullname.split('.')
        filename = self.get_ext_filename(modpath[-1])

        filename = os.path.join(*modpath[:-1]+[filename])
        return os.path.join(self.build_lib, filename)

    def get_ext_fullname(self, ext_name):
        """Returns the fullname of a given extension name.

        Adds the `package.` prefix"""
        if self.package is None:
            return ext_name
        else:
            return self.package + '.' + ext_name

    def get_ext_filename(self, ext_name):
        r"""Convert the name of an extension (eg. "foo.bar") into the name
        of the file from which it will be loaded (eg. "foo/bar.so", or
        "foo\bar.pyd").
        """
        from distutils.sysconfig import get_config_var
        ext_path = ext_name.split('.')
        ext_suffix = get_config_var('EXT_SUFFIX')
        return os.path.join(*ext_path) + ext_suffix

class CMakeExtension(Extension):
    """
    :param sourcedir: Specifies the directory (if a relative path is used, it's
                      relative to this "setup.py" file) where the "CMakeLists.txt"
                      for building the extension `name` can be found.
    """
    def __init__(self, name, sourcedir='.'):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.join(os.path.realpath('.'), sourcedir)

cmdclass['build_ext'] = CMakeBuildExt

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
)
