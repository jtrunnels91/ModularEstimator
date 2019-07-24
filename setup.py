try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()
    
setup(
    name='modest',
    version='0.1a21',
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        'pandas',
        'pint',
        'numpy',
        'matplotlib',
        'skyfield',
        'scipy',
        'pyquaternion',
        'requests',
        'astropy',
        'pyyaml>=3.12',
        'mpld3',
        'jinja2'
    ],
    description='A modular estimation library',
    long_description=long_description,
    license='MIT',
    author='Joel Runnels',
    author_email='runne010@umn.edu',
    url='https://modular-estimator.readthedocs.io/'
)
