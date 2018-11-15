import contextlib
import os
import re
import sys
import subprocess
from subprocess import CalledProcessError

from distutils.core import Command
from distutils.errors import DistutilsError

from setuptools import setup, Extension

class CMakeBuild(Command):
    description = "build C/C++ extensions using CMake"

    user_options = [
        ("defines=", "D", "Create or update cmake cache entries"),
        ("undef=", "U", "Remove matching entries from CMake cache"),
        ("generator=", "G", "Specify the build system generator"),
        ("build-temp=", 't', "Directory for temporary files (build by-products)"),
    ]

    def initialize_options(self):
        self.extensions = None
        self.package = None
        self.defines = None
        self.undef = None
        self.generator = None
        self.build_temp = None
        self.build_lib = "build"
        self.cmake_args = []
        self.build_args = []

    def finalize_options(self):
        if self.package is None:
            self.package = self.distribution.ext_package

        self.extensions = self.distribution.ext_modules

        if self.build_temp is None:
            self.build_temp = "build"

        if self.generator:
            self.cmake_args.append('-G{}'.format(self.generator))

        if self.defines:
            self.cmake_args += ['-D{}'.format(x.strip()) for x in self.defines.split(',')]

        if self.undef:
            self.cmake_args.append('-U "{}"'.format(self.undef))

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
            if not ext.optional:
                raise
            self.warn('building extension "{}" failed: {}'.format(ext.name, e))

    def build_extension(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + self.cmake_args,
                              cwd=self.build_temp, stderr=subprocess.PIPE)
        subprocess.check_call(['cmake', '--build', '.'] + self.build_args,
                              cwd=self.build_temp, stderr=subprocess.PIPE)

        fullname = self.get_ext_fullname(ext.name)
        modpath = fullname.split('.')
        filename = self.get_ext_filename(modpath[-1])
        install_dir = os.path.join(self.build_lib, *modpath[:-1])
        if not os.path.exists(self.build_lib):
            os.makedirs(install_dir)

        self.copy_file(self.get_ext_fullpath(ext.name),
                       os.path.join(install_dir, filename))

    def get_source_files(self):
        return []

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
        self.sourcedir = os.path.abspath(sourcedir)

setup(
    name='netket',
    version='0.1',
    author='Giuseppe Carleo et al.',
    description='NetKet',
    url='http://github.com/netket/netket',
    author_email='netket@netket.org',
    license='Apache',
    ext_modules=[CMakeExtension('netket')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
