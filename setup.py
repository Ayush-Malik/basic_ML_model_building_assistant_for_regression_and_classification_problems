
import sys
import os
import shutil
import glob
import subprocess
from setuptools import setup

from distutils.command.sdist import sdist


# This directory
dir_setup = os.path.dirname(os.path.realpath(__file__))

if sys.version_info < (3, 5):
    print("SymPy requires Python 3.5 or newer. Python %d.%d detected"
          % sys.version_info[:2])
    sys.exit(-1)

modules = [
    'automation.data_preprocessing',
]

tests = [
    'automation.data_preprocessing.tests',
]

with open(os.path.join(dir_setup, 'automation', 'release.py')) as f:
    # Defines __version__
    exec(f.read())


if __name__ == '__main__':
    setup(name='automation',
          version=__version__,
          description='automatically ML problem solver in Python',
          author='Ml automator team',
          author_email='Nan',
          license='MIT',
          keywords="Ml_automator",
          packages=['automation'] + modules + tests,
          ext_modules=[],
          python_requires='>=3.5',
          classifiers=[
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3',
          ],
          )
