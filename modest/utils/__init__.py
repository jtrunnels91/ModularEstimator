##
# @file __init__.py
# @brief Initialization file for the signals subpackage.

from . QuaternionHelperFunctions import euler2quaternion, quaternion2euler, eulerAngleDiff

__all__ = [
    "euler2quaternion",
    "quaternion2euler",
    "eulerAngleDiff"
]
