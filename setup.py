import os
import platform
import re
import shlex
import subprocess
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


# Poor man's command-line options parsing
def steal_cmake_flags(args):
    """
    Extracts CMake-related arguments from ``args``. ``args`` is a list of
    strings usually equal to ``sys.argv``. All arguments of the form
    ``--cmake-args=...`` are extracted (i.e. removed from ``args``!) and
    accumulated. If there are no arguments of the specified form,
    ``NETKET_CMAKE_FLAGS`` environment variable is used instead.
    """
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

"""
A list of arguments to be passed to the configuration step of CMake.
"""
_CMAKE_FLAGS = steal_cmake_flags(sys.argv)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='netket',
    version='2.0',
    author='Giuseppe Carleo et al.',
    description='NetKet',
    url='http://github.com/netket/netket',
    author_email='netket@netket.org',
    license='Apache',
    ext_modules=[CMakeExtension('netket')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
