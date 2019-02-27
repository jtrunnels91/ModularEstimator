try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name='modest',
    version='0.1.a01',
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        'Pint',
        'numpy',
        'matplotlib',
        'skyfield',
        'scipy',
        'pyquaternion',
        'requests',
        'pandas',
        'astropy',
        'PyYAML',
        'datetime',
    ],
    description='A modular estimation library',
    long_description='A modular estimation library',
    url='https://modular-estimator.readthedocs.io/en/latest/index.html',
    license='MIT',
    author='Joel Runnels',
    author_email='runne010@umn.edu'
)
