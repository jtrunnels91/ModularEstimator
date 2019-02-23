## @file __init__.py Initialization file for the modest package.
__all__ = [
    'ModularFilter',
    'substates',
    'signals',
    'utils',
    'plots',
    'spacecraft',
    'setupfunctions'
]

from . import substates
from . import signals
from . import utils
from . import spacecraft
from . import plots
from . import setupfunctions
from . modularfilter import ModularFilter

