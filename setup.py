try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name='modest',
    version='0.0.18',
    packages=find_packages(exclude=("tests",)),
	install_requires=[
            'scipy',
            'numpy',
            'math',
            'matplotlib',
            'abc',
            'pyquaternion',
            'requests',
            'pandas',
            'tempfile',
            'os',
            'subprocess',
            'astropy',
            'yaml',
            'pypet',
            'datetime',
            'warnings'
        ],
    description='A modular estimation library',
    long_description='A modular estimation library',
    author='Joel Runnels',
    author_email='runne010@umn.edu'
)
