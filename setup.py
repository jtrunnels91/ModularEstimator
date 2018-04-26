try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='modest',
    packages=['modest'],
	install_requires=['numpy','matplotlib'],
    version='0.0.2',
    description='A modular estimation library',
    author='Joel Runnels',
    author_email='runne010@umn.edu'
)
