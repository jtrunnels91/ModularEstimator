try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name='modest',
    version='0.0.25',
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'pyquaternion',
        'requests',
        'pandas',
        'astropy',
        'pyyaml',
        'datetime',
    ],
    description='A modular estimation library',
    long_description='A modular estimation library',
    license='MIT',
    author='Joel Runnels',
    author_email='runne010@umn.edu'
)
