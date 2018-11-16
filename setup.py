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
    print(cmake_args)
    return cmake_args

_CMAKE_ARGS = steal_cmake_args(sys.argv)

print(sys.argv)
print(_CMAKE_ARGS)

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

# cmdclass['build_ext'] = _CMakeBuildExt

setup(
    name='netket',
    version='0.1',
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
